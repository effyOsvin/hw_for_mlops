import torch


def get_dataloader(X, y, batch_size=64):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y).long()
    )
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    return train_loader
