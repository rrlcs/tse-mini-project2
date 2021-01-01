import torch.nn as nn


# Decoder


class FF_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(FF_Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # print("x input: ", x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc3(x))
        x = self.activation(self.fc2(x))
        # print("x output: ", x)
        return x
