'''
Code Adopted from:
https://github.com/pcyin/pytorch-gated-graph-neural-network
'''

from typing import List
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable


class AdjacencyList:
    """represent the topology of a graph"""
    def __init__(self, node_num: int, adj_list: List, device: torch.device):
        self.node_num = node_num
        self.data = torch.tensor(adj_list, dtype=torch.long, device=device)
        self.edge_num = len(adj_list)

    @property
    def device(self):
        return self.data.device

    def __getitem__(self, item):
        return self.data[item]

# Encoder


class GGNN_Encoder(nn.Module):
    def __init__(self, hidden_size, num_edge_types, layer_timesteps,
                 residual_connections,
                 state_to_message_dropout=0.3,
                 rnn_dropout=0.3,
                 use_bias_for_message_linear=True):

        super(GGNN_Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_edge_types = num_edge_types
        self.layer_timesteps = layer_timesteps
        self.residual_connections = residual_connections
        self.state_to_message_dropout = state_to_message_dropout
        self.rnn_dropout = rnn_dropout
        self.use_bias_for_message_linear = use_bias_for_message_linear

        # Prepare linear transformations from node states to messages,
        # for each layer and each edge type
        # Prepare rnn cells for each layer
        self.state_to_message_linears = []
        self.rnn_cells = []
        for layer_idx in range(len(self.layer_timesteps)):
            state_to_msg_linears_cur_layer = []
            # Initiate a linear transformation for each edge type
            for edge_type_j in range(self.num_edge_types):
                # TODO: glorot_init?
                state_to_msg_linear_layer_i_type_j = nn.Linear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=use_bias_for_message_linear
                    )
                setattr(self,
                        'state_to_message_linear_layer%d_type%d' % (
                            layer_idx,
                            edge_type_j
                            ),
                        state_to_msg_linear_layer_i_type_j)
                state_to_msg_linears_cur_layer.append(
                    state_to_msg_linear_layer_i_type_j
                    )
            self.state_to_message_linears.append(
                state_to_msg_linears_cur_layer
                )

            layer_residual_connections = self.residual_connections.get(
                layer_idx,
                []
                )
            rnn_cell_layer_i = nn.GRUCell(
                self.hidden_size * (1 + len(layer_residual_connections)),
                self.hidden_size
                )
            rnn_cell_layer_i = nn.GRUCell(self.hidden_size, self.hidden_size)
            setattr(self, 'rnn_cell_layer%d' % layer_idx, rnn_cell_layer_i)
            self.rnn_cells.append(rnn_cell_layer_i)

        self.state_to_message_dropout_layer = nn.Dropout(
            self.state_to_message_dropout
            )
        self.rnn_dropout_layer = nn.Dropout(self.rnn_dropout)

        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    @property
    def device(self):
        return self.rnn_cells[0].weight_hh.device

    def forward(self,
                initial_node_representation: Variable,
                adjacency_lists: List[AdjacencyList],
                return_all_states=False) -> Variable:

        node_representations = self.compute_node_representations(
            initial_node_representation,
            adjacency_lists,
            return_all_states=return_all_states
            )
        out = self.fc_layer(node_representations)
        graph_repr, _ = torch.max(out, dim=0)
        return graph_repr

    def compute_node_representations(self,
                                     initial_node_representation: Variable,
                                     adjacency_lists: List[AdjacencyList],
                                     return_all_states=False) -> Variable:
        # If the dimension of initial node embedding is smaller,
        # then perform padding first
        # one entry per layer (final state of that layer),
        # shape: number of nodes in batch v x D
        init_node_repr_size = initial_node_representation.size(1)
        device = adjacency_lists[0].data.device
        if init_node_repr_size < self.hidden_size:
            pad_size = self.hidden_size - init_node_repr_size
            zero_pads = torch.zeros(
                initial_node_representation.size(0),
                pad_size,
                dtype=torch.float,
                device=device
                )
            initial_node_representation = torch.cat(
                [initial_node_representation, zero_pads],
                dim=-1
                )
        node_states_per_layer = [initial_node_representation]

        node_num = initial_node_representation.size(0)

        message_targets = []  # list of tensors of message targets of shape [E]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(
                adjacency_lists
                ):
            if adjacency_list_for_edge_type.edge_num > 0:
                edge_targets = adjacency_list_for_edge_type[:, 1]
                message_targets.append(edge_targets)
        message_targets = torch.cat(message_targets, dim=0)  # Shape [M]

        # sparse matrix of shape [V, M]
        # incoming_msg_sparse_matrix =
        # self.get_incoming_message_sparse_matrix(adjacency_lists).to(device)
        for layer_idx, num_timesteps in enumerate(self.layer_timesteps):
            # Used shape abbreviations:
            #   V ~ number of nodes
            #   D ~ state dimension
            #   E ~ number of edges of current type
            #   M ~ number of messages (sum of all E)

            # Record new states for this layer. Initialised to last state,
            # but will be updated below:
            node_states_for_this_layer = node_states_per_layer[-1]
            # For each message propagation step
            for t in range(num_timesteps):
                # list of tensors of messages of shape [E, D]
                messages: List[torch.FloatTensor] = []
                # list of tensors of edge source states of shape [E, D]
                message_source_states: List[torch.FloatTensor] = []

                # Collect incoming messages per edge type
                for edge_type_idx, adjacency_list_for_edge_type in enumerate(
                        adjacency_lists
                        ):
                    if adjacency_list_for_edge_type.edge_num > 0:
                        # shape [E]
                        edge_sources = adjacency_list_for_edge_type[:, 0]
                        # shape [E, D]
                        edge_source_states = node_states_for_this_layer[
                            edge_sources
                            ]

                        f_state_to_message = self.state_to_message_linears[
                            layer_idx
                            ][
                                edge_type_idx
                            ]
                        # Shape [E, D]
                        x = self.state_to_message_dropout_layer(
                            f_state_to_message(edge_source_states)
                            )
                        all_messages_for_edge_type = x

                        messages.append(all_messages_for_edge_type)
                        message_source_states.append(edge_source_states)

                # shape [M, D]
                messages: torch.FloatTensor = torch.cat(messages, dim=0)

                # Sum up messages that go to the same target node
                # shape [V, D]
                incoming_messages = torch.zeros(
                    node_num,
                    messages.size(1),
                    device=device)
                incoming_messages = incoming_messages.scatter_add_(
                    0,
                    message_targets.unsqueeze(-1).expand_as(messages),
                    messages
                    )

                # shape [V, D * (1 + num of residual connections)]
                # incoming_information = torch.cat(layer_residual_states +
                # [incoming_messages], dim=-1)
                incoming_information = torch.cat([incoming_messages], dim=-1)

                # pass updated vertex features into RNN cell
                # Shape [V, D]
                updated_node_states = self.rnn_cells[layer_idx](
                    incoming_information,
                    node_states_for_this_layer
                    )
                updated_node_states = self.rnn_dropout_layer(
                    updated_node_states
                    )
                node_states_for_this_layer = updated_node_states

            node_states_per_layer.append(node_states_for_this_layer)

        if return_all_states:
            return node_states_per_layer[1:]
        else:
            node_states_for_last_layer = node_states_per_layer[-1]
            return node_states_for_last_layer