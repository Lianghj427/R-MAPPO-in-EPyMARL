import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RCentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(RCentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # 定义网络层：MLP -> GRU -> MLP
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # 初始化 Critic 的隐状态
        return self.fc1.weight.new(batch_size, self.n_agents, self.args.rnn_hidden_dim).zero_()

    def forward(self, batch, t=None, hidden_state=None):
        # 1. 构建输入
        inputs, bs, _ = self._build_inputs(batch, t=t)
        
        # 2. 调整维度以适配 GRUCell
        # PyMARL 的 batch[t] 通常是 [Batch, Agents, Dim] 或 [Batch*Agents, Dim]
        # 我们统一 reshape 成 [Batch * Agents, Dim]
        x = F.relu(self.fc1(inputs))
        x = x.reshape(-1, self.args.rnn_hidden_dim)

        # 3. RNN 处理
        if hidden_state is not None:
            # 确保 hidden_state 也是 [Batch * Agents, Dim]
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
        else:
            # 如果没有提供 hidden_state，比如第一步，或者非序列调用（不推荐）
            h = F.relu(x) # 降级为 MLP 行为，但通常 learner 会传入 h

        # 4. 输出 Value
        q = self.fc2(h)
        
        # 5. 还原维度
        # q: [Batch, Agents, 1]
        # h: [Batch, Agents, Dim]
        q = q.reshape(bs, self.n_agents, 1)
        h = h.reshape(bs, self.n_agents, self.args.rnn_hidden_dim)
        
        return q, h
    
    def forward_step(self, inputs, hidden_state):
        """
        专门用于 Learner 循环中的单步更新
        inputs: [Batch * N_Agents, Input_Dim] (Learner 已经 flatten 过了)
        hidden_state: [Batch * N_Agents, Hidden_Dim]
        """
        # FC1 -> ReLU
        x = F.relu(self.fc1(inputs))
        
        # Reshape for RNN if needed (though usually linear layer output matches)
        x = x.reshape(-1, self.args.rnn_hidden_dim)
        
        # GRU Cell
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
            q = self.fc2(h)
        else:
            # 理论上不应该进这里，因为 Learner 会传初始状态
            h = None
            q = self.fc2(F.relu(x))

        return q, h

    def _build_inputs(self, batch, t=None):
        # 复用 PyMARL 原有的输入构建逻辑
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        # State: [Batch, Time, State_Dim] -> 扩展到每个 Agent
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # Last Actions (可选)
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)])
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                inputs.append(last_actions)

        # Agent ID
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # 计算输入维度
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        input_shape += self.n_agents
        return input_shape