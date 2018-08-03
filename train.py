import torch
from pytoune.framework import Model, Callback
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from dataset import NameDataset
from model import NameGenerator
from utils import char_to_idx, PAD_IDX
import matplotlib.pyplot as plt


def reshape_y():
    """ For the CrossEntropyLoss we need each row to have only one label. Net already outputs each timestep in
     separate rows, so we need to do the same thing for the y"""
    def move_to_device(batch):
        x, y = default_collate(batch)
        return x, y.view(-1).long()

    return move_to_device


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
                        num_workers=0,
                        collate_fn=reshape_y())
    net = NameGenerator(len(char_to_idx))
    optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)
    criterion = CrossEntropyLoss(ignore_index=PAD_IDX)
    model = Model(net, optimizer, criterion, metrics=['accuracy']).to(device)
    history = model.fit_generator(loader, epochs=300, validation_steps=0,
                                  callbacks=[ClipGradient(net, 2), GenerateCallback(net, device)])
    plt.subplot(121)
    plt.plot([d['loss'] for d in history], label='loss')
    plt.subplot(122)
    plt.plot([d['acc'] for d in history], label='acc')
    plt.legend()
    plt.show()
