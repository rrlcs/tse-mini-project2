import torch.nn as nn


# Decoder


class FF_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(FF_Decoder, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=0.8)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # input layer --> hidden layer --> relu -->
        # dropout layer --> output layer --> sigmoid activation
        x = self.fc_in(x)
        x = self.relu(self.fc_hidden(x))
        x = self.dropout1(x)
        x = self.fc_out(x)
        x = self.activation(x)
        return x
