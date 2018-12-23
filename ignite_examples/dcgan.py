import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


class Net(nn.Module):
    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if 'Conv' in classname:
                m.weight.data.normal_(0.0, 0.02)
            elif 'BatchNorm' in classname:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        return x


class Generator(Net):
    def __init__(self, z_dim, nf):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=nf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=nf * 8, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=nf, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.weights_init()

    def forward(self, x):
        return self.net(x)


class Discriminator(Net):
    def __init__(self, nf):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nf * 4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.weights_init()

    def forward(self, x):
        output = self.net(x)
        return output.view(-1, 1).squeeze(1)


def check_dataset(dataset, dataroot):
    to_rgb = transforms.Lambda(lambda img: img.convert('RGB'))
    resize = transforms.Resize(64)
    crop = transforms.CenterCrop(64)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if dataset == 'mnist':
        dataset = dset.MNIST(root=dataroot, download=True, transform=transforms.Compose([
            to_rgb, resize, to_tensor, normalize
        ]))
    else:
        raise RuntimeError("Invalid dataset name: {}".format(dataset))

    return dataset


PRINT_FREQ = 100
FAKE_IMG_FNAME = 'fake_sample_epoch_{:04d}.png'
REAL_IMG_FNAME = 'real_sample_epoch_{:04d}.png'
LOGS_FNAME = 'logs.tsv'
PLOT_FNAME = 'plot.svg'
SAMPLES_FNAME = 'samples.svg'
CKPT_PREFIX = 'networks'

def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    z_dim = 100
    g_filters = 64
    d_filters = 64
    learning_rate = 0.0002
    beta_1 = 0.5
    alpha = 0.98
    batch_size = 64
    epochs = 25
    device = 'cuda'

    netG = Generator(z_dim, g_filters)
    netD = Discriminator(d_filters)

    bce = nn.BCELoss()

    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta_1, 0.999))

    dataset = check_dataset('mnist', '../Data/')
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    real_labels = torch.ones(batch_size, device=device)
    fake_labels = torch.zeros(batch_size, device=device)
    fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)


    def get_noise():
        return torch.randn(batch_size, z_dim, 1, 1, device=device)


    def step(engine, batch):
        real, _ = batch
        real = real.to(device)

        netD.zero_grad()

        output = netD(real)
        errD_real = bce(output, real_labels)
        D_x = output.mean().item()

        errD_real.backward()

        noise = get_noise()
        fake = netG(noise)

        output = netD(fake.detach())
        errD_fake = bce(output, fake_labels)
        D_G_z1 = output.mean().item()

        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        # GEN UPDATE
        netG.zero_grad()

        output = netD(fake)
        errG = bce(output, real_labels)
        D_G_z2 = output.mean().item()

        errG.backward()

        optimizerG.step()

        return {
            'errD': errD.item(),
            'errG': errG.item(),
            'D_x': D_x,
            'D_G_z1': D_G_z1,
            'D_G_z2': D_G_z2
        }


    output_dir = "../data/dcgan/"

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(output_dir, CKPT_PREFIX, save_interval=1, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    monitoring_metrics = ['errD', 'errG', 'D_x', 'D_G_z1', 'D_G_z2']
    RunningAverage(alpha=alpha, output_transform=lambda x: x['errD']).attach(trainer, 'errD')
    RunningAverage(alpha=alpha, output_transform=lambda x: x['errG']).attach(trainer, 'errG')
    RunningAverage(alpha=alpha, output_transform=lambda x: x['D_x']).attach(trainer, 'D_x')
    RunningAverage(alpha=alpha, output_transform=lambda x: x['D_G_z1']).attach(trainer, 'D_G_z1')
    RunningAverage(alpha=alpha, output_transform=lambda x: x['D_G_z2']).attach(trainer, 'D_G_z2')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)


    @trainer.on(Events.ITERATION_COMPLETED)
    def print_logs(engine):
        if (engine.state.iteration - 1) % PRINT_FREQ == 0:
            fname = os.path.join(output_dir, LOGS_FNAME)
            columns = engine.state.metrics.keys()
            values = [str(round(value, 5)) for value in engine.state.metrics.values()]

            with open(fname, 'a') as f:
                if f.tell() == 0:
                    print('\t'.join(columns), file=f)
                print('\t'.join(values), file=f)

            message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                                  max_epoch=epochs,
                                                                  i=(engine.state.iteration % len(loader)),
                                                                  max_i=len(loader))
            for name, value in zip(columns, values):
                message += ' | {name}: {value}'.format(name=name, value=value)

            pbar.log_message(message)


    @trainer.on(Events.EPOCH_COMPLETED)
    def save_real_example(engine):
        img, y = engine.state.batch
        path = os.path.join(output_dir, REAL_IMG_FNAME.format(engine.state.epoch))
        vutils.save_image(img, path, normalize=True)


    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={
                                  'netG': netG,
                                  'netD': netD
                              })

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)


    @trainer.on(Events.EPOCH_COMPLETED)
    def print_timer(engine):
        pbar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, timer.value()))
        timer.reset()


    @trainer.on(Events.EPOCH_COMPLETED)
    def create_plots(engine):
        try:
            import matplotlib as mpl
            mpl.use('agg')

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

        except ImportError:
            warnings.warn('Loss plots will not be generated -- pandas or matplotlib not found')

        else:
            df = pd.read_csv(os.path.join(output_dir, LOGS_FNAME), delimiter='\t')
            x = np.arange(1, engine.state.iteration + 1, PRINT_FREQ)
            _ = df.plot(x=x, subplots=True, figsize=(20, 20))
            _ = plt.xlabel('Iteration number')
            fig = plt.gcf()
            path = os.path.join(output_dir, PLOT_FNAME)

            fig.savefig(path)


    # @trainer.on(Events.EXCEPTION_RAISED)
    # def handle_exception(engine, e):
    #     if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
    #         engine.terminate()
    #         warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
    #
    #         create_plots(engine)
    #         checkpoint_handler(engine, {
    #             'netG_exception': netG,
    #             'netD_exception': netD
    #         })
    #
    #     else:
    #         raise e

    trainer.run(loader, epochs)


main()
