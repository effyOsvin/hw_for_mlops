import torch


def train_epoch(model, optimizer, train_loader, criterion, device):
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        x_pred = model(x)

        # train_step
        loss = criterion(x_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_loss_acc(loader, model, criterion, device):
    model.eval()
    loss, accuracy = 0.0, 0.0
    batches_count = 0
    full_count = 0
    with torch.no_grad():
        for x, y in loader:
            full_count += len(y)
            batches_count += 1
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss += criterion(y_pred, y).item()
            accuracy += torch.sum(torch.argmax(y_pred, dim=-1) == y)
    accuracy = float(accuracy) / full_count
    loss = loss / batches_count
    return loss, accuracy


def train_model(
    model, opt, train_loader, test_loader, criterion, n_epochs, device, verbose=True
):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []

    for epoch in range(n_epochs):
        train_epoch(model, opt, train_loader, criterion, device)
        train_loss, train_acc = evaluate_loss_acc(train_loader, model, criterion, device)
        val_loss, val_acc = evaluate_loss_acc(test_loader, model, criterion, device)

        train_log.append(train_loss)
        train_acc_log.append(train_acc)

        val_log.append(val_loss)
        val_acc_log.append(val_acc)

        if verbose:
            print(
                (
                    "Epoch [%d/%d], Loss (train/val): %.4f/%.4f,"
                    + " Acc (train/val): %.4f/%.4f"
                )
                % (epoch + 1, n_epochs, train_loss, val_loss, train_acc, val_acc)
            )

    return train_log, train_acc_log, val_log, val_acc_log


def save_all(model, model_parameters, save_name):
    model_dict = model.state_dict()
    tmp_save = [model_dict, model_parameters]
    torch.save(tmp_save, save_name)
