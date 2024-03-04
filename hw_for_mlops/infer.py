import sys

import pandas as pd
import torch
import torch.utils.data
from tools.get_dataloader import get_dataloader
from tools.infer_model import load_model, validate
from tools.load_data import load_data


def infer():
    model_name = "best_model.xyz"
    save_name = "test_results.csv"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        if len(sys.argv) > 2:
            save_name = sys.argv[2]
    X_test, y_test = load_data(train=False)
    test_loader = get_dataloader(X_test, y_test)

    model = load_model(model_name)
    device = torch.device("cpu")
    model = model.to(device)
    model_answer, accuracy = validate(model, test_loader, device)
    data = pd.DataFrame(model_answer)
    data.to_csv(save_name, sep="\t", encoding="utf-8")
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    infer()
