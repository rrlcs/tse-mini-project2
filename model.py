from ggnn_encoder import GGNN_Encoder
from ffnn_decoder import FF_Decoder
from ggnn2ff import GGNN2FF
import torch
import numpy as np

# Training Hyperparamters
learning_rate = 1e-5

# Model Hyperparameters
load_model = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 128
output_size = 29
hidden_size = 128


# Weight initialization
# Code borrowed from: https://stackoverflow.com/a/55546528/10439780
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)


# Initializing the encoder
encoder_net = GGNN_Encoder(
    hidden_size=hidden_size,
    num_edge_types=1,
    layer_timesteps=[5]
    ).to(device)
# Initializing the decoder
decoder_net = FF_Decoder(input_size, output_size, hidden_size).to(device)
# Initializing the model with defined encoder and decoder
model = GGNN2FF(encoder_net, decoder_net).to(device)
model.apply(weights_init_normal)
