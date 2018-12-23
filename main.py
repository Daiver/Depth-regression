import os
import torch
import matplotlib.pyplot as plt

from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite import metrics

from dataset import NyuDepthDataset
import paths
import models


train_dataset, test_dataset = NyuDepthDataset.from_file_with_splits(
    os.path.join(paths.data_root, 'nyu_depth_v2_labeled.mat'), os.path.join(paths.data_root, 'splits.mat'))
print("N train {}, N test {}".format(len(train_dataset), len(test_dataset)))

model = models.LinkNet(1, 3, True)


