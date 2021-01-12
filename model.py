from ggnn_encoder import GGNN_Encoder
from ffnn_decoder import FF_Decoder
from ggnn2ff import GGNN2FF
import torch

# Training Hyperparamters
learning_rate = 1e-5

# Model Hyperparameters
load_model = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 128
output_size = 29
hidden_size = 128

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
