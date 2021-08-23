import torch.nn as nn
import torch
import numpy as np
from sklearn import preprocessing
from code.ggnn_encoder import AdjacencyList
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GGNN2FF: Combining Encoder Decoder


class GGNN2FF(nn.Module):
    def __init__(self, encoder, decoder):
        super(GGNN2FF, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, edge_list, feature, num_of_nodes, target):
        target_len = target.shape[0]
        # Scaling features to standard normal distribution
        scalar = preprocessing.StandardScaler().fit(feature)
        feature_scaled = torch.from_numpy(
            np.array(scalar.transform(feature))
            ).to(dtype=torch.float16)
        # initialize output vector
        output = torch.zeros(target_len, 1).to(device)
        # Obtain Adjacency list from edge list of AST
        adj_list = AdjacencyList(
            node_num=num_of_nodes,
            adj_list=edge_list,
            device=device
            )
        # Obtain graph representation from GGNN Encoder
        graph_repr = self.encoder(feature_scaled, adjacency_lists=[adj_list])
        # Reshaping graph_repr to (hidden_size * 1)
        graph_repr = graph_repr.reshape((1, -1)).unsqueeze(0)
        # Output vector of size (No. of grammar rules * 1)
        # Contains probabilities for the corresponding,
        # grammar rule to be present
        output = self.decoder(graph_repr)
        return output
