class lstm_char(nn.Module):
    def __init__(self, unique_input_tokens, inputsize, hiddensize, outputsize, num_layer, dropout_prob=0.2):
        super().__init__()

        self.hiddensize = hiddensize
        self.num_layer = num_layer
        self.int2char = dict(enumerate(unique_input_tokens))
        self.char2int = {char: int for (int, char) in int2char.items()}

        self.lstm = nn.LSTM(input_size=inputsize,
                            hidden_size=hiddensize,
                            num_layers=num_layer,
                            batch_first=True,  # since we are using batches of data we set this to true!
                            dropout=dropout_prob)

        self.dropout = nn.Dropout2d(dropout_prob)
        self.fc = nn.Linear(hiddensize, outputsize)

    def forward(self, x, hidden):
        """
        hidden contains hiddenstate and cellstate
        """
        output, hidden = self.lstm(x, hidden)

        output = self.dropout(output)
        output = output.contiguous().view(-1, self.hiddensize)
        output = self.fc(output)
        return output, hidden
lstm_char()