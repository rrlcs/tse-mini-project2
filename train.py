from model import model
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import math
import re
import numpy as np
from torch.optim.lr_scheduler import StepLR


# Set parameters for training
epochs = 50
learning_rate = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_size = 35456

# Define criterion
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=9e-5
    )

# Learning Rate scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

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
def save_checkpoint(state, filename="training_checkpoint.path.ptor"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


# Format data in correct size and shape and type
def format_data(
    feature_matrix_for_ast_nodes,
    num_of_nodes,
    edge_list,
    rules_used
):
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
    return feature_matrix_for_ast_nodes, edge_list_of_tuples, rules_used


# Start Trainig
model.train()
start = time.time()
print_loss_total = 0
loss_list_epoch = []
for epoch in range(1, epochs+1):
    loss_list = []  # contains loss for each iteration
    for i in range(1, training_size+1):

        # Extract data elements from the training data
        feature_matrix_for_ast_nodes = training_data[i].get('featureMatrix')
        num_of_nodes = training_data[i].get('num_of_nodes')
        edge_list = training_data[i].get('edgeList')
        rules_used = training_data[i].get('GrammarRulesUsed')

        # Get data in correct format
        feature_matrix_for_ast_nodes, edge_list_of_tuples, rules_used = format_data(
            feature_matrix_for_ast_nodes,
            num_of_nodes,
            edge_list,
            rules_used
            )

        # Run the model to get output vector
        output = model(
            edge_list_of_tuples,
            feature_matrix_for_ast_nodes,
            num_of_nodes,
            rules_used
            )

        optimizer.zero_grad()

        # Reshape output according to rules_used
        output = output.squeeze(0).permute(1, 0)

        # Calculate loss
        loss = criterion(output, rules_used)

        # Print losses and save checkpoints every 500 iterations
        print_loss_total += loss
        if i % 500 == 0:

            # Compute average loss every 500 iterations
            print_loss_avg = print_loss_total / 500

            # Append average loss to the loss_list
            loss_list.append(print_loss_avg)

            # Store sum of losses for every 500 iterations
            f1 = open("sum_lossess_every_500", "a")
            f1.write(str(print_loss_total))
            f1.close

            # Re-initialize total loss to zero
            print_loss_total = 0

            # Print loss, iteration and time remaining
            print("loss: ", loss, "iteration: ", i)
            print('%s (%d %d%%) %.4f' % (timeSince(
                start,
                i / len(training_data)
                ), i, i / len(training_data) * 100, print_loss_avg))

            # Get model checkpoint
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }

            # Save checkpoint
            save_checkpoint(checkpoint)

            # Store losses in the file
            f1 = open("lossess", "a")
            f1.write(str(loss_list))
            f1.close

            # Re-initialize the loss_list to empty
            loss_list = []

        # Propagate loss backward
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15)

        # Update weights
        optimizer.step()

    # Decrement learnig rate to 1/10th every epoch
    scheduler.step()

    # Print learning rate
    print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

    # Compute total sum of losses over complete training set
    with open("sum_lossess_every_500", "r") as f:
        data = f.read()
    data = re.findall("\d+\.\d+", data)
    data = np.array(list(map(float, data)))

    # Compute loss for each epoch
    loss_list_epoch = data.sum() / (training_size * epoch)

    # Store loss per epoch in file
    f2 = open("lossess_per_epoch", "a")
    f2.write(str(loss_list_epoch)+",")
    f2.close
    loss_list_epoch = []
