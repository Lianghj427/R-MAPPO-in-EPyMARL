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

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        return self.fc1.weight.new(batch_size, self.n_agents, self.args.rnn_hidden_dim).zero_()

    def forward(self, batch, t=None, hidden_state=None):
        inputs, bs, _ = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = x.reshape(-1, self.args.rnn_hidden_dim)

        if hidden_state is not None:
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = F.relu(x)

        q = self.fc2(h)
        
        q = q.reshape(bs, self.n_agents, 1)
        h = h.reshape(bs, self.n_agents, self.args.rnn_hidden_dim)
        
        return q, h
    
    def forward_step(self, inputs, hidden_state):

        x = F.relu(self.fc1(inputs))
        
        x = x.reshape(-1, self.args.rnn_hidden_dim)
        
        # GRU Cell
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
            q = self.fc2(h)
        else:
            h = None
            q = self.fc2(F.relu(x))

        return q, h

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

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
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        input_shape += self.n_agents
        return input_shape