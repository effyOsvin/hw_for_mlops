import numpy as np
import torch
from hw_for_mlops.models.model import ConvLinear


def load_model(filename):
    [model_state_dict, model_parameters] = torch.load(filename)
    model = ConvLinear()
    model.load_state_dict(model_state_dict)
    return model


def validate(model, test_loader, device):
    model.eval()
    accuracy = 0.0
    batches_count = 0
    counter = 0
    model_answer = []
    with torch.no_grad():
        for x, y in test_loader:
            counter += len(y)
            batches_count += 1
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            model_answer.append((np.argmax(y_pred.cpu().numpy(), axis=-1)))
            accuracy += torch.sum(torch.argmax(y_pred, dim=-1) == y)
    accuracy = float(accuracy) / counter
    model_answer = np.hstack(model_answer)
    return model_answer, accuracy
