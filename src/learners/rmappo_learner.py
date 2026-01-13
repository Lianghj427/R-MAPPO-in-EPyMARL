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
        self.value_loss_coef = getattr(args, "value_loss_coef", 1)
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
        avail_actions = batch["avail_actions"][:, :-1]
        alive_mask = batch["alive_agents"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]

        last_actions_onehot = th.zeros_like(actions_onehot)
        last_actions_onehot[:, 1:] = actions_onehot[:, :-1]
        
        # Prepare storage for rollout results
        batch_size = batch.batch_size
        max_seq_length = batch.max_seq_length - 1 # Exclude last step for training data
        
        # We need to collect:
        # - old_action_log_probs
        # - values
        # - actor_hidden_states (for chunk initialization)
        # - critic_hidden_states (if critic is RNN)
        
        old_action_log_probs = []
        values = []
        actor_hidden_states = []
        critic_hidden_states = []
        
        self.mac.init_hidden(batch_size)
        # Initialize critic hidden state if it has one (assuming it mimics MAC interface if RNN)
        c_hidden = None
        if hasattr(self.critic, "init_hidden"):
             c_hidden = self.critic.init_hidden(batch_size)

        # Forward pass over the entire episode to get hidden states and old probs
        with th.no_grad():
            for t in range(max_seq_length + 1): # Go one step further for value of next state
                # Actor
                actor_h = self.mac.hidden_states.clone() # Store state BEFORE update
                actor_hidden_states.append(actor_h)
                
                agent_outs = self.mac.forward(batch, t=t)
                
                # Calculate log probs
                # Note: This simple calculation works for Discrete actions. 
                # For more complex output types, use action_selector logic.
                pi = agent_outs
                pi_taken = th.gather(pi, dim=2, index=batch["actions"][:, t])
                log_pi = th.log(pi_taken + 1e-10)
                if t < max_seq_length:
                    old_action_log_probs.append(log_pi)

                # Critic
                # Handling RNN critic if present
                if c_hidden is not None:
                    critic_hidden_states.append(c_hidden.clone())
                    v, c_hidden = self.critic.forward(batch, t=t, hidden_state=c_hidden)
                else:
                    v = self.critic.forward(batch, t=t) # Standard MLP critic
                    
                values.append(v)

        # Stack results
        old_action_log_probs = th.stack(old_action_log_probs, dim=1).squeeze(-1) # [B, T, N]
        values = th.stack(values, dim=1) # [B, T+1, N, 1]
        actor_hidden_states = th.stack(actor_hidden_states, dim=1) # [B, T+1, N, Dim]
        
        if len(critic_hidden_states) > 0:
            critic_hidden_states = th.stack(critic_hidden_states, dim=1)
        
        # ----------------------------------------------------------------------
        # 2. CALCULATE ADVANTAGES (GAE)
        # ----------------------------------------------------------------------
        next_value = values[:, -1]
        values = values[:, :-1]
        
        returns, advantages = self._compute_gae(rewards, mask, values, next_value)
        
        # Normalize advantages
        if self.is_normalize_advantages:
            sum_advantages = (advantages * alive_mask).sum()
            count_advantages = alive_mask.sum()
            mean_advantages = sum_advantages / (count_advantages + 1e-8)
            var_advantages = ((advantages - mean_advantages) ** 2 * alive_mask).sum() / (count_advantages + 1e-8)
            std_advantages = th.sqrt(var_advantages)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        # ----------------------------------------------------------------------
        # 3. PPO UPDATE LOOP (With Data Chunking)
        # ----------------------------------------------------------------------
        
        # Data structure to pass to generator
        # Flatten batch and time dimensions initially for easier handling in generator
        episode_data = {
            "obs": batch["obs"][:, :-1],
            "state": batch["state"][:, :-1],
            "actions": actions,
            "avail_actions": avail_actions,
            "old_action_log_probs": old_action_log_probs,
            "returns": returns,
            "values": values,
            "advantages": advantages,
            "mask": mask,
            "alive_mask": alive_mask,
            "actor_hidden_states": actor_hidden_states[:, :-1], # Only need start of steps
            "actions_onehot": actions_onehot,
            "last_actions_onehot": last_actions_onehot
        }

        if len(critic_hidden_states) > 0:
            episode_data["critic_hidden_states"] = critic_hidden_states[:, :-1]

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

    def _compute_gae(self, rewards, mask, values, next_value):
        """ Calculate Generalized Advantage Estimation """
        gae = 0
        
        returns = th.zeros_like(values)
        advantages = th.zeros_like(values)
        
        # Check normalization
        if self.use_valuenorm:
            values_pred = self.value_normalizer.denormalize(values)
            next_value_pred = self.value_normalizer.denormalize(next_value.unsqueeze(1)).squeeze(1)
        else:
            values_pred = values
            next_value_pred = next_value

        # Reverse iteration
        for t in reversed(range(rewards.size(1))):
            if t == rewards.size(1) - 1:
                next_val = next_value_pred
            else:
                next_val = values_pred[:, t + 1]
            reward_t = rewards[:, t].unsqueeze(1) 
            mask_t = mask[:, t].unsqueeze(1)
            
            delta = reward_t + self.args.gamma * next_val * mask_t - values_pred[:, t]
            gae = delta + self.args.gamma * self.args.gae_lambda * mask_t * gae
            
            advantages[:, t] = gae
            returns[:, t] = gae + values_pred[:, t]
            
        return returns, advantages

    def _generate_data_chunks(self, data, mini_batch_size, chunk_length):
        """ 
        Simulates MAPPO's recurrent generator. 
        Splits episodes into chunks of length `chunk_length`.
        Shuffles chunks and yields mini-batches.
        """
        # data values are [Batch, Time, Agents, ...]
        batch_size, max_seq_len = data["advantages"].shape[0], data["advantages"].shape[1]
        n_agents = data["advantages"].shape[2]
        
        num_chunks_per_episode = max_seq_len // chunk_length
        total_chunks = batch_size * num_chunks_per_episode
        
        # Indices to shuffle
        rand_perm = th.randperm(total_chunks).tolist()
        
        
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
            
            yield batch_sample

    def _ppo_update(self, sample, train_info):
        # Unpack sample
        obs = sample["obs"]
        state = sample["state"]
        actions = sample["actions"]
        log_probs_old = sample["old_action_log_probs"]
        returns = sample["returns"]
        values_old = sample["values"]
        advantages = sample["advantages"]
        mask = sample["mask"]
        alive_mask = sample["alive_mask"]
        avail_actions = sample["avail_actions"]
        actor_h_start = sample["actor_hidden_states"]

        actions_onehot = sample["actions_onehot"]
        last_actions_onehot = sample["last_actions_onehot"]
        
        B, T, N, _ = obs.shape

        init_h = actor_h_start[:, 0, :, :].contiguous()
        
        # We assume BasicMAC has a way to accept hidden states or we hack it.
        # BasicMAC usually does: self.hidden_states = ...
        self.mac.hidden_states = init_h

        critic_h = None
        if "critic_hidden_states" in sample:
            critic_h_start = sample["critic_hidden_states"]
            critic_h = critic_h_start[:, 0, :, :].reshape(-1, self.args.rnn_hidden_dim).contiguous()
        
        new_log_probs = []
        dist_entropy = []
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
            
            agent_outs = agent_outs.view(B, N, -1)
            curr_avail = avail_actions[:, t]

            agent_outs[curr_avail == 0] = -1e10
            probs = F.softmax(agent_outs, dim=-1)
            
            pi_taken = th.gather(probs, dim=2, index=actions[:, t])
            log_pi = th.log(pi_taken + 1e-10)
            entropy = -th.sum(probs * th.log(probs + 1e-10), dim=-1)
            
            new_log_probs.append(log_pi)
            dist_entropy.append(entropy)

            critic_inputs = self._build_critic_inputs_from_tensors(
                state[:, t], obs[:, t], last_actions_onehot[:, t], B, N
            )

            if hasattr(self.critic, "forward_step"):
                c_q, critic_h = self.critic.forward_step(critic_inputs, critic_h)
            else:
                c_q = self.critic(critic_inputs)
                
            c_q = c_q.reshape(B, N, 1)
            new_values.append(c_q)

        new_log_probs = (th.stack(new_log_probs, dim=1)) # [B, T, N, 1]
        dist_entropy = th.stack(dist_entropy, dim=1)
        dist_entropy = (dist_entropy * mask).sum() / (mask.sum() + 1e-8)
        new_values = th.stack(new_values, dim=1) # [B, T, N, 1]
        
        # ----------------------------------------
        # Loss Calculation
        # ----------------------------------------
        
        # Policy Loss
        ratio = th.exp(new_log_probs - log_probs_old.unsqueeze(-1))
        surr1 = ratio * advantages
        surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        # Masked Policy Loss
        policy_loss = -th.sum(th.min(surr1, surr2) * alive_mask) / (alive_mask.sum() + 1e-8)
        
        # Value Loss
        if self.use_valuenorm:
            self.value_normalizer.update(returns)
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

        if self.use_clipped_value_loss:
            value_loss = th.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
            
        value_loss = (value_loss * alive_mask).sum() / (alive_mask.sum() + 1e-8)

        # Total Loss
        loss = policy_loss - (dist_entropy * self.entropy_coef) + (value_loss * self.value_loss_coef)

        # Optimize
        self.agent_optimiser.zero_grad()
        self.critic_optimiser.zero_grad()
        loss.backward()
        
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.max_grad_norm)
        c_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
        
        self.agent_optimiser.step()
        self.critic_optimiser.step()

        # Update Stats
        train_info["value_loss"] += value_loss.item()
        train_info["policy_loss"] += policy_loss.item()
        train_info["dist_entropy"] += dist_entropy.item()
        train_info["actor_grad_norm"] += grad_norm.item()
        train_info["critic_grad_norm"] += c_grad_norm.item()
        train_info["ratio"] += ratio.mean().item()

    def _build_inputs_from_tensors(self, obs_t, last_action_onehot_t, B, N):
        inputs = []
        inputs.append(obs_t) # [B, N, ObsDim]
        
        if self.args.obs_last_action:
            inputs.append(last_action_onehot_t) 
                
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=self.device).unsqueeze(0).expand(B, -1, -1))
            
        inputs = th.cat([x.reshape(B*N, -1) for x in inputs], dim=1)
        return inputs

    def _build_critic_inputs_from_tensors(self, state_t, obs_t, last_action_t, B, N):

        inputs = []
        
        # 1. State: [B, StateDim] -> [B, N, StateDim]
        state_repeated = state_t.unsqueeze(1).expand(B, N, -1)
        inputs.append(state_repeated)

        # 2. Obs (Optional)
        if getattr(self.args, "obs_individual_obs", False):
            inputs.append(obs_t) # [B, N, ObsDim]

        # 3. Last Action (Optional)
        if getattr(self.args, "obs_last_action", False):
            inputs.append(last_action_t) # [B, N, ActDim]

        # 4. Agent ID
        # [B, N, N]
        inputs.append(th.eye(N, device=self.device).unsqueeze(0).expand(B, -1, -1))
        
        # Concat & Flatten -> [B*N, TotalDim]
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
