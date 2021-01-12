import torch
import json
import time
import math
import torch.nn as nn
import numpy as np
from model import model
import torch.optim as optim
from sklearn.metrics import accuracy_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define criterion
criterion = nn.MSELoss()


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


# Load saved checkpoint
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Set parameters for loading checkpoints
learning_rate = 1e-5
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=9e-5
    )

# Load saved checkpoint
load_checkpoint(
    torch.load('training_checkpoint_5.path.ptor'),
    model,
    optimizer
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


# Start Evaluation
model.eval()
start = time.time()
acc = 0  # Initialize accuracy with 0
print_loss_total = 0
loss_list_epoch = []
loss_list = []
start_test_index = 25921
end_test_index = 30000
with torch.no_grad():
    for i in range(start_test_index, end_test_index):

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

        # Reshape output
        output = output.squeeze(0).permute(1, 0)

        # Compute loss
        loss = criterion(output, rules_used)

        # Total loss for every iterations
        print_loss_total += loss

        # Print loss, iteration and time every 500 iterations
        if i % 500 == 0:

            # Compute average loss for every 500 iteratins
            print_loss_avg = print_loss_total / 500

            # Append average losses to the loss_list
            loss_list.append(print_loss_avg)

            # Store the test sum of losses every 500 to a file
            f1 = open("test_sum_lossess_every_500_5", "a")
            f1.write(str(print_loss_total))
            f1.close

            # Re-initialize total loss to 0
            print_loss_total = 0

            # print loss and iteration
            print("loss: ", loss, "iteration: ", i)

            # print time remaining
            print('%s (%d %d%%) %.4f' % (timeSince(
                start,
                i / len(training_data)
                ), i, i / len(training_data) * 100, print_loss_avg))

            # Store test losses to file
            f1 = open("test_lossess_5", "a")
            f1.write(str(loss_list))
            f1.close

            # Re-initialize the loss_list
            loss_list = []

        # Round of the output vector to get final prediction
        predictions = torch.round(output).squeeze(0)

        # Compute the test accuracy
        acc += accuracy_score(rules_used, predictions)

    # Print the final accuracy on test data
    print("Accuracy: ", acc / (end_test_index - start_test_index))
