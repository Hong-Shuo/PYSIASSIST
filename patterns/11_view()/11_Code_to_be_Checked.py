class TextClassificationRNN(nn.Module):
    def __init__(self, batch, vocab_size, w2v_dim, hidden_size, num_layers, num_classes):
        super(TextClassificationRNN, self).__init__()

        self.batch = batch
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.w2v_dim = w2v_dim

        self.emb = nn.Embedding(self.vocab_size, self.w2v_dim)
        self.rnn = nn.LSTM(self.w2v_dim, self.hidden_size, self.num_layers)
        self.init_state = self.init_hidden()
        self.dense = nn.Linear(self.hidden_size, int(self.hidden_size / 2))
        self.dense2class = nn.Linear(int(self.hidden_size / 2), 2)
        self.dropout = nn.Dropout(0.5)

    def init_hidden(self):
        h0 = torch.randn(self.num_layers, self.batch, self.hidden_size, device=device)
        c0 = torch.randn(self.num_layers, self.batch, self.hidden_size, device=device)
        return (h0, c0)

    def forward(self, x, seq_len_list):
        x = self.emb(x)
        x = x.view(x.shape[1], x.shape[0], -1)
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(x, seq_len_list, enforce_sorted=False)
        output, hn = self.rnn(packed_data, self.init_state)
        unpacked_data, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        dense_output = self.dense(unpacked_data[-1])
        dense_output = self.dropout(dense_output)
        dense2class = self.dense2class(dense_output)
        return dense2class


objective = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
model = TextClassificationRNN(128, encoder.vocab_size, 200, 256, 1, 2)
if torch.cuda.is_available():
    model.cuda()