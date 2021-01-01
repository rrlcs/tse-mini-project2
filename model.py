from ggnn_encoder import GGNN_Encoder
from ffnn_decoder import FF_Decoder
from ggnn2ff import GGNN2FF
import torch

# Training Hyperparamters
learning_rate = 1e-4

# Model Hyperparameters
load_model = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 256
output_size = 29
hidden_size = 256

# Initializing the model

encoder_net = GGNN_Encoder(
    hidden_size,
    num_edge_types=1,
    layer_timesteps=[20, 20],
    residual_connections={2: [0], 3: [0, 1]}
    ).to(device)
decoder_net = FF_Decoder(input_size, output_size, hidden_size).to(device)

model = GGNN2FF(encoder_net, decoder_net).to(device)
