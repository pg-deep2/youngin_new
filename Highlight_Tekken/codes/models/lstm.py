import torch.nn as nn
import torch

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class bi_LSTM(nn.Module):
    def __init__(self, n_classes=2, input_size=512 * 3 * 9 * 9, hidden_size=2, n_layers=1):
        super(bi_LSTM, self).__init__()

        self.n_classes = n_classes
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    # input size = [n_snippet, 512 * 3 * 9 * 9]
    # sequence length = n_snippet
    def forward(self, x):
        x = x.unsqueeze(0) # [1, n_snippets=sequence length, 512 * 3 * 9 * 9]

        n_batch = x.size(0)

        h_0 = torch.zeros(self.n_layers*2, n_batch, self.hidden_size).to(device) # 2 for bidirectional
        c_0 = torch.zeros(self.n_layers*2, n_batch, self.hidden_size).to(device) # 2 for bidirectional

        # Propagate input through lstm
        out, _ = self.lstm(x, (h_0, c_0))
        return out.view(-1, self.n_classes)  # one hot: [Pr(0), Pr(1)]