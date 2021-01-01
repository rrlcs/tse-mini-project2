from model import model
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import math
import numpy as np
# from data import JsonDataset
# from torch.utils.data import DataLoader

epochs = 10
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-15
    )

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

# dataset = JsonDataset(["dataset.json"])
# dataloader = DataLoader(dataset, batch_size=32)


# Time Measures

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# Checkpoints


def save_checkpoint(state, filename="my_checkpoint.path.ptor"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


# Trainig

model.train()
start = time.time()
print_loss_total = 0
loss_list_epoch = []
for epoch in range(epochs):
    loss_list = []
    for i in range(1, 40000):
        # Extract data elements from the training data
        feature_matrix_for_ast_nodes = training_data[i].get('featureMatrix')
        num_of_nodes = training_data[i].get('num_of_nodes')
        edge_list = training_data[i].get('edgeList')
        rules_used = training_data[i].get('GrammarRulesUsed')

        feature_matrix_for_ast_nodes = torch.tensor(
            np.reshape(
                np.array(
                    feature_matrix_for_ast_nodes
                    ),
                (num_of_nodes, len(ast_node_encoding)+1)
                    ),
            dtype=torch.float).to(device)
        edge_list_of_tuples = list(map(tuple, edge_list))
        rules_used = torch.tensor(
            np.reshape(np.array(rules_used), (len(grammar_rules), 1)),
            dtype=torch.float).to(device)
        output = model(
            edge_list_of_tuples,
            feature_matrix_for_ast_nodes,
            num_of_nodes,
            rules_used
            )
        optimizer.zero_grad()
        output = output.squeeze(0).permute(1, 0)
        loss = criterion(output, rules_used)
        print_loss_total += loss
        if i % 500 == 0:
            print_loss_avg = print_loss_total / 500
            loss_list.append(print_loss_avg)
            print_loss_total = 0
            print("loss: ", loss, "iteration: ", i)
            print('%s (%d %d%%) %.4f' % (timeSince(
                start,
                i / len(training_data)
                ), i, i / len(training_data) * 100, print_loss_avg))
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            save_checkpoint(checkpoint)
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40)
        optimizer.step()
    loss_list_epoch.append(sum(loss_list))
