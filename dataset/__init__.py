import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision.transforms import Compose

from utils import char_to_idx, PAD_IDX, EOF_IDX


class CharToIndex:
    def __init__(self, char_to_idx, append_eof, eof_idx):
        self.char_to_idx = char_to_idx
        self.append_eof = append_eof
        self.eof_idx = eof_idx

    def __call__(self, string: str):
        suffix = [self.eof_idx] if self.append_eof else []
        return np.array([self.char_to_idx[c] for c in string] + suffix)


class PadToSize:
    def __init__(self, size, value, append_length):
        self.size = size
        self.value = value
        self.append_length = append_length

    def __call__(self, index_array):
        padded = np.pad(index_array, [0, self.size - len(index_array)], mode='constant', constant_values=self.value)
        if self.append_length:
            padded = np.hstack([padded, len(index_array)])
        return padded


class ToTensor():

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray):
        return torch.tensor(x)


class NameDataset(Dataset):
    def __init__(self, path: str):
        self.data = self._read_csv(path)
        self.max_len = self._max_len()
        self.transformer_x = Compose((CharToIndex(char_to_idx, False, EOF_IDX),
                                      PadToSize(self.max_len, PAD_IDX, True),
                                      ToTensor()))
        self.transformer_y = Compose((CharToIndex(char_to_idx, True, EOF_IDX),
                                      PadToSize(self.max_len, PAD_IDX, False),
                                      ToTensor()))

    def _max_len(self):
        return self.data['name'].str.len().max() + 1  # + 1 because EOF

    def _read_csv(self, path: str):
        data = pd.read_csv(path, header=None, names=['name'])
        data['name'] = data['name'].str.strip().str.lower()
        return data

    def __getitem__(self, index):
        name = self.data.iloc[index]['name']
        x = self.transformer_x(name)
        y = self.transformer_y(name[1:])
        return x, y

    def __len__(self):
        return len(self.data)
