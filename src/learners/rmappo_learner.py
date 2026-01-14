import copy
import numpy as np
import torch as th
from torch.optim import Adam
import torch.nn.functional as F
from components.episode_buffer import EpisodeBatch
from modules.critics import REGISTRY as critic_registry
from components.standarize_stream import RunningMeanStd

class RMAPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        
        # PPO Hyperparameters
        self.clip_param = args.eps_clip
        self.ppo_epoch = args.epochs
        self.mini_batch_size = getattr(args, "mini_batch_size", 64) # Chunk size if using chunks
        self.data_chunk_length = getattr(args, "data_chunk_length", 10)
        self.entropy_coef = getattr(args, "entropy_coef", 0.001)
        self.max_grad_norm = getattr(args, "grad_norm_clip", 0.5)
        self.use_clipped_value_loss = getattr(args, "use_clipped_value_loss", True)
        self.use_huber_loss = getattr(args, "use_huber_loss", False)
        self.huber_delta = getattr(args, "huber_delta", 10.0)
        self.is_normalize_advantages = getattr(args, "is_normalize_advantages", False)
        self.use_valuenorm = getattr(args, "use_valuenorm", True)

        # Agent Optimiser
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        # Critic
        self.critic = critic_registry[args.critic_type](scheme, args)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        # Value Normalizer
        if self.use_valuenorm:
            self.value_normalizer = RunningMeanStd(shape=(1,), device=self.args.device)
        else:
            self.value_normalizer = None

        self.device = self.args.device
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # ----------------------------------------------------------------------
        # 1. PREPARE DATA & ROLLOUT (Re-infer hidden states for chunking)
        # ----------------------------------------------------------------------
        # Get raw data from buffer
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()

        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        avail_actions = batch["avail_actions"][:, :-1]
        alive_mask = batch["alive_agents"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]

        last_actions_onehot = th.zeros_like(actions_onehot)
        last_actions_onehot[:, 1:] = actions_onehot[:, :-1]
        
        # Prepare storage for rollout results
        batch_size = batch.batch_size
        max_seq_length = batch.max_seq_length - 1 # Exclude last step for training data
        
        old_mac_out = []
        values = []
        actor_hidden_states = []
        critic_hidden_states = []

        if self.args.standardise_rewards:
            expanded_reward = rewards.repeat(1, 1, self.n_agents).unsqueeze(-1)
            sum_rewards = (expanded_reward*alive_mask).sum()
            count_rewards = alive_mask.sum()
            mean_rewards = sum_rewards / (count_rewards + 1e-8)
            var_rewards = (((expanded_reward - mean_rewards) ** 2) * alive_mask).sum() / (count_rewards + 1e-8)
            std_rewards = th.sqrt(var_rewards)
            rewards = (rewards - mean_rewards) / (std_rewards + 1e-8)
        
        self.mac.init_hidden(batch_size)

        # Initialize critic hidden state if it has one (assuming it mimics MAC interface if RNN)
        c_hidden = None
        if hasattr(self.critic, "init_hidden"):
             c_hidden = self.critic.init_hidden(batch_size)

        # Forward pass over the entire episode to get hidden states and old probs
        with th.no_grad():
            for t in range(max_seq_length):
                # Actor
                # This can be stroed during roollout, but i won't use it here
                actor_h = self.mac.hidden_states.clone()
                actor_hidden_states.append(actor_h)
                agent_outs = self.mac.forward(batch, t=t)
                old_mac_out.append(agent_outs)

                # Critic
                # Handling RNN critic if present
                if c_hidden is not None:
                    critic_hidden_states.append(c_hidden.clone())
                    v, c_hidden = self.critic.forward(batch, t=t, hidden_state=c_hidden)
                else:
                    v = self.critic.forward(batch, t=t) # Standard MLP critic
                values.append(v)

        # Stack results
        old_mac_out = th.stack(old_mac_out, dim=1) # [B, T, N]
        old_values = th.stack(values, dim=1) # [B, T+1, N, 1]
        actor_hidden_states = th.stack(actor_hidden_states, dim=1) # [B, T+1, N, Dim]
        if len(critic_hidden_states) > 0:
            critic_hidden_states = th.stack(critic_hidden_states, dim=1)
        
        # Get old action log probs
        old_pi = old_mac_out
        old_pi[alive_mask.squeeze(-1) == 0] = 1.0
        old_action_log_probs = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_action_log_probs = th.log(old_action_log_probs + 1e-10)
        
        # ----------------------------------------------------------------------
        # 2. CALCULATE ADVANTAGES (GAE) and RETURNS
        # ----------------------------------------------------------------------
        old_values = old_values.squeeze(-1).detach()
        old_values = th.cat(
            (old_values, th.zeros_like(old_values[:, 0:1, ...]),), dim=1
        )
        returns, advantages = self._compute_returns_advs(
            old_values, 
            rewards.repeat(1, 1, self.n_agents), 
            terminated.repeat(1, 1, self.n_agents),
            self.args.gamma, 
            self.args.tau
            )
        
        # Normalize advantages
        if self.is_normalize_advantages:
            sum_advantages = (advantages * alive_mask.squeeze(-1)).sum()
            count_advantages = alive_mask.sum()
            mean_advantages = sum_advantages / (count_advantages + 1e-8)
            var_advantages = (((advantages - mean_advantages) ** 2) * alive_mask.squeeze(-1)).sum() / (count_advantages + 1e-8)
            std_advantages = th.sqrt(var_advantages)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        # ----------------------------------------------------------------------
        # 3. PPO UPDATE LOOP (With Data Chunking)
        # ----------------------------------------------------------------------

        episode_data = {
            "obs": batch["obs"][:, :-1],
            "state": batch["state"][:, :-1],
            "actions": actions,
            "avail_actions": avail_actions,
            "old_action_log_probs": old_action_log_probs,
            "returns": returns,
            "values_old": old_values[:, :-1],
            "advantages": advantages,
            "mask": mask,
            "alive_mask": alive_mask.squeeze(-1),
            "actor_hidden_states": actor_hidden_states,
            "actions_onehot": actions_onehot,
            "last_actions_onehot": last_actions_onehot
        }

        if len(critic_hidden_states) > 0:
            episode_data["critic_hidden_states"] = critic_hidden_states

        train_info = {
            "value_loss": 0,
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "critic_grad_norm": 0,
            "ratio": 0
        }

        num_updates = 0

        for _ in range(self.ppo_epoch):
            # Generator handles shuffling and chunking
            data_generator = self._generate_data_chunks(
                episode_data, 
                self.mini_batch_size, 
                self.data_chunk_length
            )

            for sample in data_generator:
                self._ppo_update(sample, train_info)
                num_updates += 1

        # Average logs
        for k in train_info.keys():
            train_info[k] /= num_updates

        # Log
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("value_loss", train_info["value_loss"], t_env)
            self.logger.log_stat("policy_loss", train_info["policy_loss"], t_env)
            self.logger.log_stat("dist_entropy", train_info["dist_entropy"], t_env)
            self.logger.log_stat("grad_norm", train_info["actor_grad_norm"], t_env)
            if self.is_normalize_advantages:
                self.logger.log_stat("advantage_mean", mean_advantages.item(), t_env)
            self.log_stats_t = t_env

    def _compute_returns_advs(
        self, _values, _rewards, _terminated, gamma, tau
    ):
        returns = th.zeros_like(_rewards)
        advs = th.zeros_like(_rewards)
        lastgaelam = th.zeros_like(_rewards[:, 0]).flatten()
        ts = _rewards.size(1)

        for t in reversed(range(ts)):
            nextnonterminal = (1 - _terminated[:, t]).flatten()
            nextvalues = _values[:, t + 1].flatten()

            reward_t = _rewards[:, t].flatten()
            delta = (
                reward_t
                + gamma * nextvalues * nextnonterminal
                - _values[:, t].flatten()
            )
            lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam

            advs[:, t] = lastgaelam.view(_rewards[:, t].shape)

        returns = advs + _values[:, :-1]

        return returns, advs

    def _generate_data_chunks(self, data, mini_batch_size, chunk_length):
        """ 
        Simulates MAPPO's recurrent generator. 
        Splits episodes into chunks of length `chunk_length`.
        Shuffles chunks and yields mini-batches.
        """
        # data values are [Batch, Time, Agents, ...]
        batch_size, max_seq_len = data["advantages"].shape[0], data["advantages"].shape[1]
        n_agents = data["advantages"].shape[2]
        
        # Calculate how many full chunks we can make
        # Note: If max_seq_len is not divisible by chunk_length, we might drop the end or pad.
        # MAPPO usually assumes fixed length episodes or proper masking.
        num_chunks_per_episode = max_seq_len // chunk_length
        total_chunks = batch_size * num_chunks_per_episode
        
        # Indices to shuffle
        rand_perm = th.randperm(total_chunks).tolist()
        
        # Flat dictionary to store chunked data
        # We need to reshape everything to [Total_Chunks, Chunk_Len, Agents, ...]
        
        # Helper to slice and stack
        def slice_chunks(tensor):
            # tensor: [Batch, Time, ...]
            # reshape to [Batch, Num_Chunks, Chunk_Len, ...]
            shape = tensor.shape
            dims = shape[2:]
            
            # Truncate to fit chunks
            limit = num_chunks_per_episode * chunk_length
            truncated = tensor[:, :limit]
            
            chunked = truncated.reshape(batch_size, num_chunks_per_episode, chunk_length, *dims)
            # Flatten batch and chunks -> [Total_Chunks, Chunk_Len, ...]
            return chunked.reshape(total_chunks, chunk_length, *dims)

        chunked_data = {k: slice_chunks(v) for k, v in data.items()}

        # Yield mini-batches
        num_minibatches = total_chunks // mini_batch_size
        
        for i in range(num_minibatches):
            indices = rand_perm[i * mini_batch_size : (i+1) * mini_batch_size]
            
            batch_sample = {}
            for k, v in chunked_data.items():
                batch_sample[k] = v[indices] # [MiniBatch, Chunk_Len, ...]
            
            # For RNNs, we usually want [Chunk_Len, MiniBatch, Agents, ...] (Time-major)
            # But PyMARL modules usually expect [Batch, Time, ...].
            # However, for the RNN update, we specifically need to handle the hidden state.
            
            yield batch_sample

    def _ppo_update(self, sample, train_info):
        # Unpack sample
        obs = sample["obs"]
        state = sample["state"]
        actions = sample["actions"]
        log_probs_old = sample["old_action_log_probs"]
        returns = sample["returns"].unsqueeze(-1)
        values_old = sample["values_old"].unsqueeze(-1)
        advantages = sample["advantages"].unsqueeze(-1)
        mask = sample["mask"]
        alive_mask = sample["alive_mask"]
        avail_actions = sample["avail_actions"]
        actor_h_start = sample["actor_hidden_states"]

        actions_onehot = sample["actions_onehot"]
        last_actions_onehot = sample["last_actions_onehot"]
        
        B, T, N, _ = obs.shape
        
        # ----------------------------------------
        # Re-evaluate Policy and Value on Chunks
        # ----------------------------------------
        
        # We need to run the MAC on this chunk.
        # Crucial: Initialize MAC hidden state with the saved start state!
        # actor_h_start is [B, 1, N, Dim] -> we need [B, N, Dim]
        init_h = actor_h_start[:, 0, :, :].contiguous()
        
        self.mac.hidden_states = init_h

        critic_h = None
        if "critic_hidden_states" in sample:
            critic_h_start = sample["critic_hidden_states"]
            critic_h = critic_h_start[:, 0, :, :].reshape(-1, self.args.rnn_hidden_dim).contiguous()
        
        new_mac_out = []
        new_values = []
        
        # Rollout loop for the chunk
        for t in range(T):
            last_action_input = last_actions_onehot[:, t]
            agent_inputs = self._build_inputs_from_tensors(
                obs[:, t], 
                last_action_input, # Logic needs care for t=0 of chunk
                B, N
            )
            alive_agent = alive_mask[:, t]
            
            agent_outs, self.mac.hidden_states = self.mac.agent(agent_inputs, self.mac.hidden_states, alive_agent)
            
            # Reshape for computation
            agent_outs = agent_outs.view(B, N, -1)
            if getattr(self.args, "mask_before_softmax", True):
                curr_avail = avail_actions[:, t]
                agent_outs[curr_avail == 0] = -1e10
            probs = F.softmax(agent_outs, dim=-1)

            new_mac_out.append(probs)

            critic_inputs = self._build_critic_inputs_from_tensors(
                state[:, t],
                last_actions_onehot[:, t],
                B, 
                N
            )

            c_q, critic_h = self.critic.forward_step(critic_inputs, critic_h)
            c_q = c_q.reshape(B, N, 1)
            new_values.append(c_q)

        new_mac_out = th.stack(new_mac_out, dim=1)
        new_values = th.stack(new_values, dim=1)

        # Get old action log probs
        new_pi = new_mac_out
        new_pi[alive_mask == 0] = 1.0
        log_probs_new = th.gather(new_pi, dim=3, index=actions).squeeze(3)
        log_probs_new = th.log(log_probs_new + 1e-10)
        
        # ----------------------------------------
        # Loss Calculation
        # ----------------------------------------

        # Entropy loss
        dist_entropy = -th.sum(new_pi * th.log(new_pi + 1e-10), dim=-1)
        dist_entropy = (dist_entropy * alive_mask).sum() / (alive_mask.sum() + 1e-8)
        
        # Policy Loss
        ratio = th.exp(log_probs_new.unsqueeze(-1) - log_probs_old.unsqueeze(-1))
        surr1 = ratio * advantages
        surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        # Masked Policy Loss
        policy_loss = th.sum(th.min(surr1, surr2) * alive_mask.unsqueeze(-1)) / (alive_mask.sum() + 1e-8)
        
        # Value Loss
        if self.use_valuenorm:
            self.value_normalizer.update(returns, mask=alive_mask)
            returns_normalized = self.value_normalizer.normalize(returns)
            values_pred_clipped = values_old + (new_values - values_old).clamp(-self.clip_param, self.clip_param)
            error_clipped = returns_normalized - values_pred_clipped
            error_original = returns_normalized - new_values
        else:
            values_pred_clipped = values_old + (new_values - values_old).clamp(-self.clip_param, self.clip_param)
            error_clipped = returns - values_pred_clipped
            error_original = returns - new_values

        if self.use_huber_loss:
            target_values = returns_normalized if self.use_valuenorm else returns
            value_loss_clipped = F.huber_loss(values_pred_clipped, target_values, delta=self.huber_delta, reduction='none')
            value_loss_original = F.huber_loss(new_values, target_values, delta=self.huber_delta, reduction='none')
        else:
            value_loss_clipped = error_clipped ** 2
            value_loss_original = error_original ** 2

        # Critic: value loss
        value_loss = th.max(value_loss_original, value_loss_clipped) if self.use_clipped_value_loss else value_loss_original
        value_loss = (value_loss * alive_mask.unsqueeze(-1)).sum() / (alive_mask.sum() + 1e-8)

        # Actor: policy gradient Loss
        pg_loss = -(policy_loss + (dist_entropy * self.entropy_coef))

        # Optimize
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        actor_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.max_grad_norm)
        self.agent_optimiser.step()

        self.critic_optimiser.zero_grad()
        value_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
        self.critic_optimiser.step()

        # Update Stats
        train_info["value_loss"] += value_loss.item()
        train_info["policy_loss"] += policy_loss.item()
        train_info["dist_entropy"] += dist_entropy.item()
        train_info["actor_grad_norm"] += actor_grad_norm.item()
        train_info["critic_grad_norm"] += critic_grad_norm.item()
        train_info["ratio"] += ((ratio*alive_mask.unsqueeze(-1)).sum() / (alive_mask.sum() + 1e-8)).item()

    def _build_inputs_from_tensors(self, obs_t, last_action_onehot_t, B, N):
        inputs = []
        inputs.append(obs_t) # [B, N, ObsDim]
        
        if self.args.obs_last_action:
            inputs.append(last_action_onehot_t) 
                
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=self.device).unsqueeze(0).expand(B, -1, -1))
            
        inputs = th.cat([x.reshape(B*N, -1) for x in inputs], dim=1)
        return inputs

    def _build_critic_inputs_from_tensors(self, state_t, last_action_t, B, N):
        inputs = []
        
        state_repeated = state_t.unsqueeze(1).expand(B, N, -1)
        inputs.append(state_repeated)

        if getattr(self.args, "obs_last_action", False):
            inputs.append(last_action_t) # [B, N, ActDim]

        inputs.append(th.eye(N, device=self.device).unsqueeze(0).expand(B, -1, -1))
        
        inputs = th.cat(inputs, dim=-1)
        return inputs.reshape(B*N, -1)
    
    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.agent_optimiser.state_dict(), "{}/actor_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
