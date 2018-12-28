import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from dataset import StringToPaddedIndexesWithLength
from utils import char_to_idx, idx_to_char, PAD_IDX, EOS


class NameGenerator(nn.Module):
    def __init__(self, input_size, hidden_size=64, lstm_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.embedding = nn.Embedding(input_size + 1, hidden_size, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_with_length):
        x, _ = self._forward(x_with_length)
        return x

    def _forward(self, x_with_length, lstm_state=None):
        x = x_with_length[:, :-1]
        lengths = x_with_length[:, -1]
        total_padded_length = x.size(-1)
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, lstm_state = self.lstm(x, lstm_state)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=PAD_IDX, total_length=total_padded_length)
        x = self.linear(x)
        x = x.view(-1, x.size(-1))  # Each timestamp in the separate row, so that we can pass it to CrossEntropyLoss
        return x, lstm_state

    def generate(self, device, max_len=40):
        transformer = StringToPaddedIndexesWithLength(max_len)
        i = 0
        c = np.random.choice(list(char_to_idx.keys() - ['', EOS]))
        out = ''
        hidden_size = (self.lstm_layers, 1, self.hidden_size)
        h_n, c_n = torch.zeros(*hidden_size, device=device), torch.zeros(*hidden_size, device=device)
        while c != EOS and i < max_len:
            x = transformer(c).unsqueeze(0).to(device)
            x, (h_n, c_n) = self._forward(x, (h_n, c_n))
            x = x[0]  # when generating we always have only one timestamp(letter) and it's the first one
            distribution = Categorical(logits=x)
            c = idx_to_char[distribution.sample().item()]
            out += c
            i += 1
        return out
