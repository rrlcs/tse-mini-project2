import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


# class JsonDataset(IterableDataset):
#     def __init__(self, files):
#         self.files = files

#     def __iter__(self):
#         for json_file in self.files:
#             with open(json_file) as f:
#                 for sample_line in f:
#                     sample = json.loads(sample_line)
#                     # print("sam: ", sample["constraint"])
#                     yield sample["constraint"], sample["program"]


# Grammar Rules Dictionary
grammar_rules = dict(
    [
        ('var1', 1), ('var2', 2), ('num', 3), ('plus', 4),
        ('tplus', 5), ('uminus', 6), ('minus', 7), ('mul1', 8),
        ('mul2', 9), ('div', 10), ('mod', 11), ('ite', 12),
        ('true', 13), ('false', 14), ('and', 15), ('or', 16),
        ('implies', 17), ('xor', 18), ('xnor', 19), ('nand', 20),
        ('nor', 21), ('iff', 22), ('not', 23), ('beq', 24),
        ('leq', 25), ('geq', 26), ('lt', 27), ('gt', 28),
        ('eq', 29)
        ]
    )

# AST Node Encoding into one-hot vector Dictionary
ast_node_encoding = dict(
    {
        'f': 0, 'var1': 1, 'var2': 2, '=': 3, '<': 4, '>': 5,
        '<=': 6, '>=': 7, 'and': 8, 'or': 9, 'not': 10,
        '=>': 12, '+': 13, '-': 14, '*': 15, 'div': 16,
        'PAD1': 17, 'PAD2': 18, 'PAD3': 19
        }
                          )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("dataset.json", "r") as f:
    data = json.load(f)
dataset = data.get('TrainingExamples')

# features_list = []
# num_of_nodes_list = []
# edge_list_list = []
# rules_used_list = []
# for i in range(len(dataset)):
#     # Extract data elements from the training data
#     feature_matrix_for_ast_nodes = dataset[i].get('featureMatrix')
#     num_of_nodes = dataset[i].get('num_of_nodes')
#     edge_list = dataset[i].get('edgeList')
#     rules_used = dataset[i].get('GrammarRulesUsed')
#     np.reshape(
#         np.array(
#             feature_matrix_for_ast_nodes
#             ), (num_of_nodes, len(ast_node_encoding)+1)
#         )
#     np.reshape(np.array(rules_used), (len(grammar_rules), 1))
#     # feature_matrix_for_ast_nodes = torch.tensor(
#     #     np.reshape(
#     #         np.array(
#     #             feature_matrix_for_ast_nodes
#     #             ),
#     #         (num_of_nodes, len(ast_node_encoding)+1)
#     #             ),
#     #     dtype=torch.float).to(device)
#     edge_list_of_tuples = list(map(tuple, edge_list))
#     # rules_used = torch.tensor(
#     #     np.reshape(np.array(rules_used), (len(grammar_rules), 1)),
#     #     dtype=torch.float).to(device)

#     features_list.append(feature_matrix_for_ast_nodes)
#     num_of_nodes_list.append(num_of_nodes)
#     edge_list_list.append(edge_list_of_tuples)
#     rules_used_list.append(rules_used)

# print(np.reshape(np.array(features_list), )
# features_tensor = torch.tensor(features_list)
# num_of_nodes_tensor = torch.tensor(num_of_nodes_list)
# edge_list_tensor = torch.tensor(edge_list_list)
# rules_used_tensor = torch.tensor(rules_used_list)
# dataset = TensorDataset(
#     features_list,
#     num_of_nodes_list,
#     edge_list_list,
#     rules_used_list
#     )

dataframe = pd.DataFrame.from_dict(
    dataset
    )
# print(dataframe['edgeList'])
print(len(dataset))
for i in range(len(dataset)):
    dataframe['featureMatrix'][i] = torch.tensor(np.reshape(
        np.array(
            dataframe['featureMatrix'][i]
        ),
        (
            dataframe['num_of_nodes'][i],
            len(ast_node_encoding)+1
        )
    ))
    dataframe['edgeList'][i] = torch.tensor(
        np.array(
            list(map(tuple, dataframe['edgeList'][i]))
        )
    )
    dataframe['GrammarRulesUsed'][i] = torch.tensor(np.reshape(
        np.array(
            dataframe['GrammarRulesUsed'][i]
        ),
        (
            len(grammar_rules),
            1
        )
    ))
    dataframe['num_of_nodes'][i] = torch.tensor(np.reshape(
        np.array(
            dataframe['num_of_nodes'][i]
        ),
        (1, 1)
    ))

print((dataframe['featureMatrix'].values).size())
print(dataframe['edgeList'].values.size())
print(dataframe['GrammarRulesUsed'].size())
print(dataframe['num_of_nodes'].values.size())
# a = dataframe['featureMatrix'].values
# print(a)
dataset = TensorDataset(torch.tensor(np.array([1, 2]))
    # dataframe['featureMatrix'].values,
    # dataframe['edgeList'].values,
    # dataframe['GrammarRulesUsed'].values,
    # dataframe['num_of_nodes'].values
    )
# print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, )
# dataloader.batch_sampler()
# print(dataloader._iterator()._next_data())
# print(dataloader.dataset.__getitem__(10))
# for index, item in enumerate(dataloader):
#     print(index, len(item))
#     print(item)
