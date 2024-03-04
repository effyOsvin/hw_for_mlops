import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from models.model import ConvLinear
from tools.get_dataloader import get_dataloader
from tools.load_data import load_data
from tools.train_model import save_all, train_model


def train():
    n_epochs = 3
    save_name = "best_model.xyz"
    if len(sys.argv) > 1:
        n_epochs = int(sys.argv[1])
        if len(sys.argv) > 2:
            save_name = sys.argv[2]

    X_train, y_train = load_data(train=True)

    idxs = np.random.permutation(np.arange(X_train.shape[0]))
    X_train, y_train = X_train[idxs], y_train[idxs]

    train_loader = get_dataloader(X_train[:25000], y_train[:25000])
    val_loader = get_dataloader(X_train[25000:30000], y_train[25000:30000])

    model_parameters = {"p": 0.23}
    model = ConvLinear(model_parameters["p"])
    opt = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    train_model(
        model, opt, train_loader, val_loader, criterion, n_epochs, device, verbose=True
    )

    save_all(model, model_parameters, save_name)


if __name__ == "__main__":
    train()
