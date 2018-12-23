import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from tqdm import tqdm
from ignite.engine import Engine, create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss


import numpy as np


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7*7*64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7*7*64)
        x = self.fc1(x)

        return F.log_softmax(x, dim=-1)


batch_size = 64
lr = 1e-3
train_loader, val_loader = get_data_loaders(batch_size, batch_size)

model = ConvNet()
device = 'cuda'
optimizer = optim.Adam(model.parameters(), lr=lr)
trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
evaluator = create_supervised_evaluator(
    model=model,
    metrics={'accuracy': Accuracy(),
             'nll': Loss(F.nll_loss)},
    device=device
)

desc = "ITERATION - loss: {:.2f}"
pbar = tqdm(
    initial=0, leave=False, total=len(train_loader),
    desc=desc.format(0)
)


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    pbar.desc = desc.format(engine.state.output)
    pbar.update(1)


@trainer.on(Events.EPOCH_COMPLETED)
def log_train_metrics(engine):
    pbar.refresh()
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    tqdm.write(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_nll))


@trainer.on(Events.EPOCH_COMPLETED)
def log_val_metrics(engine):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    tqdm.write(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, metrics['accuracy'], metrics['nll']))

    pbar.n = pbar.last_print_n = 0


trainer.run(train_loader, max_epochs=10)
pbar.close()
