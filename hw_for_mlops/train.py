import sys

import torch
import torch.nn as nn
import torch.utils.data
from models.model import ConvLinear
from sklearn.model_selection import train_test_split
from tools.get_dataloader import get_dataloader
from tools.load_data import load_data
from tools.train_model import save_all, train_model


def train():
    n_epochs = 3
    save_name = "bin/best_model.xyz"
    if len(sys.argv) > 1:
        n_epochs = int(sys.argv[1])
        if len(sys.argv) > 2:
            save_name = sys.argv[2]

    X, y = load_data(train=True)
    count_data = X.shape[0] // 2
    X_train, X_val, y_train, y_val = train_test_split(
        X[:count_data], y[:count_data], test_size=0.2, random_state=48
    )
    train_loader = get_dataloader(X_train, y_train)
    val_loader = get_dataloader(X_train, y_train)

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
