import torch.nn as nn


# Decoder


class FF_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(FF_Decoder, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.7)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # print("x input: ", x)
        x = self.fc_in(x)
        x = self.relu(self.fc_hidden(x))
        x = self.dropout1(x)
        x = self.relu(self.fc_hidden(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        x = self.activation(x)
        # print("x output: ", x)
        return x
