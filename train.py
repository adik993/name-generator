import pandas as pd

import matplotlib.pyplot as plt
import torch
from pytoune.framework import Model, Callback
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import NameDataset
from model import NameGenerator
from utils import char_to_idx, PAD_IDX


class ClipGradient(Callback):
    def __init__(self, module: NameGenerator, clip):
        super().__init__()
        self.clip = clip
        self.module = module

    def on_batch_end(self, batch, logs):
        torch.nn.utils.clip_grad_value_(self.module.parameters(), self.clip)


class GenerateCallback(Callback):
    def __init__(self, net: NameGenerator, device, n=10, every=10):
        super().__init__()
        self.net = net
        self.device = device
        self.n = n
        self.every = every

    def on_epoch_end(self, epoch, logs):
        if epoch % self.every == 0:
            self.net.train(False)
            print('\n'.join([self.net.generate(self.device) for i in range(self.n)]))
            self.net.train(True)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NameDataset('./data/data.csv')
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=4,
                        collate_fn=dataset.sort_by_length_flatten_on_timestamp_collate)
    net = NameGenerator(len(char_to_idx))
    optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)
    criterion = CrossEntropyLoss(ignore_index=PAD_IDX)
    model = Model(net, optimizer, criterion, metrics=['accuracy']).to(device)
    history = model.fit_generator(loader, epochs=300, validation_steps=0,
                                  callbacks=[ClipGradient(net, 2), GenerateCallback(net, device)])
    df = pd.DataFrame(history).set_index('epoch')
    df.plot(subplots=True)
    plt.show()
