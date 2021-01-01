import torch.nn as nn
import torch
from ggnn_encoder import AdjacencyList
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GGNN2FF: Combining Encoder Decoder


class GGNN2FF(nn.Module):
    def __init__(self, encoder, decoder):
        super(GGNN2FF, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, edge_list, feature, num_of_nodes, target):
        target_len = target.shape[0]

        output = torch.zeros(target_len, 1).to(device)

        adj_list = AdjacencyList(
            node_num=num_of_nodes,
            adj_list=edge_list,
            device=device
            )

        graph_repr = self.encoder(feature, adjacency_lists=[adj_list])

        graph_repr = graph_repr.reshape((1, -1)).unsqueeze(0)

        output = self.decoder(graph_repr)
        return output
