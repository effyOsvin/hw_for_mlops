import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tools.get_dataloader import get_dataloader
from tools.infer_model import load_model, validate
from tools.load_data import load_data


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def infer(cfg: DictConfig):
    X_test, y_test = load_data(cfg.test.X_path, cfg.test.y_path)
    test_loader = get_dataloader(np.array(X_test), np.array(y_test))

    model = load_model(cfg.model.save_name)
    model = model.to(cfg.test.device)
    model_answer, accuracy = validate(model, test_loader, cfg.test.device)
    data = pd.DataFrame(model_answer)
    data.to_csv(cfg.test.save_name, sep="\t", encoding="utf-8")
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    infer()
