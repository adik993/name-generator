import torch
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


class ToTensor:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray):
        return torch.tensor(x)


class StringToPaddedIndexesWithLength:
    def __init__(self, max_len):
        self.transformer = Compose((CharToIndex(char_to_idx, False, EOF_IDX),
                                    PadToSize(max_len, PAD_IDX, True),
                                    ToTensor()))

    def __call__(self, x: str):
        return self.transformer(x)


class StringToPaddedIndexesWithEosAppended:
    def __init__(self, max_len):
        self.transformer = Compose((CharToIndex(char_to_idx, True, EOF_IDX),
                                    PadToSize(max_len, PAD_IDX, False),
                                    ToTensor()))

    def __call__(self, x: str):
        return self.transformer(x)
