import torch.nn as nn
import torch.nn.functional as F

class RNNLayer(nn.Module):
    def __init__(self, input_shape, config):
        super(RNNLayer, self).__init__()
        
        self.hidden_dim = config['rnn_hidden_dim']
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

    def init_hidden(self, hidden_size):
        # make hidden states on same device as model
        return self.fc1.weight.new(hidden_size, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        return h