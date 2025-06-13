
import torch.nn as nn 

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNLayer(nn.Module):
    
    def __init__(self, config, obs_shape):
        
        super().__init__()
        self.hidden_size = config['cnn_hidden_size']
        self.kernel_size = config['cnn_kernel_size']
        self.stride = config['cnn_stride']
        use_ReLU = config['cnn_use_ReLU']
        use_orthogonal = config['cnn_use_orthogonal']
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])
        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        cnn_out_size = self.hidden_size // 2 * (input_width - self.kernel_size + self.stride) * (input_height - self.kernel_size + self.stride)
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=self.hidden_size // 2,
                            kernel_size=self.kernel_size,
                            stride=self.stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(cnn_out_size,
                            self.hidden_size)),
                            active_func,
                            )
    def forward(self, input):
        output = self.cnn(input)
        return output
