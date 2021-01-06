import torch
import json
import numpy as np
from model import model
import torch.optim as optim
from sklearn.metrics import accuracy_score
# from train import grammar_rules, ast_node_encoding, training_data, device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


learning_rate = 1e-5
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-10
    )
load_checkpoint(torch.load('my_checkpoint.path.ptor'), model, optimizer)
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


# Loading Data

with open("dataset.json", "r") as f:
    data = json.load(f)
training_data = data.get('TrainingExamples')


model.eval()
acc = 0
with torch.no_grad():
    for i in range(25921, 30000):
        feature_matrix_for_ast_nodes = training_data[i].get('featureMatrix')
        num_of_nodes = training_data[i].get('num_of_nodes')
        edge_list = training_data[i].get('edgeList')
        rules_used = training_data[i].get('GrammarRulesUsed')

        feature_matrix_for_ast_nodes = torch.tensor(
            np.reshape(
                np.array(feature_matrix_for_ast_nodes),
                (num_of_nodes, len(ast_node_encoding)+1)),
            dtype=torch.float
            ).to(device)
        edge_list_of_tuples = list(map(tuple, edge_list))
        rules_used = torch.tensor(
            np.reshape(
                np.array(rules_used),
                (len(grammar_rules), 1)),
            dtype=torch.float
            ).to(device)
        output = model(
            edge_list_of_tuples,
            feature_matrix_for_ast_nodes,
            num_of_nodes,
            rules_used
            )
        # output = nn.Sigmoid()(output)

        print("output: ", output)
        predictions = torch.round(output).squeeze(0).permute(1, 0)
        print(predictions.size())
        print(rules_used.size())
        acc += accuracy_score(rules_used, predictions)
    print("Accuracy: ", acc / 4079)
    # print("rules_used: ", rules_used)
