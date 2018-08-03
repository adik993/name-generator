import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

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
        x = x_with_length[:, :-1]
        total_padded_lenght = x.size(-1)
        lengths = x_with_length[:, -1]
        x = self.embedding(x)

        sorted_lengths, sorted_idx = torch.sort(lengths, dim=0, descending=True)
        sorted_x = x[sorted_idx]
        x = pack_padded_sequence(sorted_x, sorted_lengths, batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=PAD_IDX, total_length=total_padded_lenght)
        x = torch.zeros_like(x).scatter_(0, sorted_idx.unsqueeze(1).unsqueeze(1).expand(-1, x.shape[1], x.shape[2]), x)

        x = self.linear(x)
        return x.view(-1, x.size(-1))  # Each timestamp in the separate row, so that we can pass it to CrossEntropyLoss

    def generate(self, device, max_len=40):
        i = 0
        c = np.random.choice(list(char_to_idx.keys() - ['']))
        out = ''
        hidden_size = (self.lstm_layers, 1, self.hidden_size)
        h_n, c_n = torch.zeros(*hidden_size, device=device), torch.zeros(*hidden_size, device=device)
        while c != EOS and i < max_len:
            x = torch.tensor([[char_to_idx[c]]], device=device, dtype=torch.long)
            x = self.embedding(x)
            x, (h_n, c_n) = self.lstm(x, (h_n, c_n))
            x = self.linear(x)
            distribution = Categorical(logits=x.view(-1))
            c = idx_to_char[distribution.sample().item()]
            out += c
            i += 1
        return out
