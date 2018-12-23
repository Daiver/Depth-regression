import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor

from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite import metrics
from ignite.contrib.handlers import ProgressBar

from dataset import NyuDepthDataset
import paths
import models


def dual_compose(transforms):
    composed = Compose(transforms)

    def inner(img, depth):
        return composed(img), composed(depth)
    return inner


def main():
    epochs = 100
    batch_size = 4
    lr = 1e-3
    device = 'cuda'

    model = models.LinkNet(1, 3, True)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    evaluator = create_supervised_evaluator(model, {"loss": metrics.Loss(F.mse_loss)}, device)

    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_val_metrics(engine):
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        evaluator.run(test_loader)
        val_metrics = evaluator.state.metrics
        print(engine.state.epoch, train_metrics, val_metrics)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model_state(engine):
        fname = "model_{}.pt".format(engine.state.epoch)
        torch.save(model, fname)

    train_dataset, test_dataset = NyuDepthDataset.from_file_with_splits(
        os.path.join(paths.data_root, 'nyu_depth_v2_labeled.mat'), os.path.join(paths.data_root, 'splits.mat'),
        transform=dual_compose([ToTensor()])
    )
    # for x, y in train_dataset:
    #     print(x.shape, y.shape, type(x), type(y))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("N train {}, N test {}".format(len(train_dataset), len(test_dataset)))

    print('Start')
    state = trainer.run(train_loader, epochs)
    print(state.output)
    print('End')


main()
