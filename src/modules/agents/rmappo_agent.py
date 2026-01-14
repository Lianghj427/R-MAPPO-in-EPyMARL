import torch.nn as nn
import torch.nn.functional as F
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class RMAPPOAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RMAPPOAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):

            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=0.01)

            nn.init.orthogonal_(self.rnn.weight_ih, gain=1.0)
            nn.init.orthogonal_(self.rnn.weight_hh, gain=1.0)
            
            if self.rnn.bias_ih is not None:
                nn.init.constant_(self.rnn.bias_ih, 0)
            if self.rnn.bias_hh is not None:
                nn.init.constant_(self.rnn.bias_hh, 0)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, alive_agent):
        inputs = inputs.view(-1, self.args.n_agents, self.input_shape)
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b*a, -1), hh.view(b, a, -1)