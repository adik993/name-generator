import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from dataset.transformers import StringToPaddedIndexesWithLength, StringToPaddedIndexesWithEosAppended


class NameDataset(Dataset):
    def __init__(self, path: str):
        self.data = self._read_csv(path)
        self.max_len = self._max_len()
        self.transformer_x = StringToPaddedIndexesWithLength(self.max_len)
        self.transformer_y = StringToPaddedIndexesWithEosAppended(self.max_len)

    def _max_len(self):
        return self.data['name'].str.len().max()

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

    @staticmethod
    def sort_by_length_flatten_on_timestamp_collate(batch):
        """
         This must be used as collate_fn when creating DataLoader out of this dataset.
         It does two things:
          * sorts x and y by lengths stored as last column in x so that we can pass it to pack_padded_sequence
          * flattens y along timestamp dimension. For the CrossEntropyLoss we need each row to have only one label.
            Net already outputs each timestep in separate rows, so we need to do the same thing for the y"""
        x_with_lengths, y = default_collate(batch)
        lengths = x_with_lengths[:, -1]
        sorted_indexes = torch.argsort(lengths, dim=0, descending=True)
        return x_with_lengths[sorted_indexes], y[sorted_indexes].view(-1).long()
