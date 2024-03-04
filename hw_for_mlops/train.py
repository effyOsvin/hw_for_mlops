import hydra
import torch
import torch.nn as nn
import torch.utils.data
from models.model import ConvLinear
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from tools.get_dataloader import get_dataloader
from tools.load_data import load_data
from tools.train_model import save_all, train_model


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    X, y = load_data(cfg.train.X_path, cfg.train.y_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X[: cfg.train.count_data],
        y[: cfg.train.count_data],
        test_size=cfg.train.test_size,
        random_state=cfg.train.random_state,
    )
    train_loader = get_dataloader(X_train, y_train)
    val_loader = get_dataloader(X_train, y_train)

    model_parameters = {"dropout": cfg.model.dropout, "out_dim": cfg.model.out_dim}
    model = ConvLinear(cfg.model.dropout, cfg.model.out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()
    train_model(
        model,
        opt,
        train_loader,
        val_loader,
        criterion,
        cfg.train.num_epoch,
        cfg.train.device,
        verbose=True,
    )

    save_all(model, model_parameters, cfg.model.save_name)


if __name__ == "__main__":
    train()
