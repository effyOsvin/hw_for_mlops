import json

import hydra
import numpy as np
import requests
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def run_server(cfg: DictConfig):
    url = cfg.infer.tracking_uri + "/invocations"
    X = np.random.rand(*cfg.sample.shape)
    data = {"inputs": X.tolist()}
    headers = {"content-type": "application/json"}
    request = requests.post(url, json=data, headers=headers)
    if request.status_code == 200:
        predict = request.text
        jsonString = json.dumps(predict)
        jsonFile = open(cfg.infer.result_path, "w")
        jsonFile.write(jsonString)
        jsonFile.close()
        print(f"Success! Predictions in {cfg.infer.result_path}")
    else:
        print("Something wrong(((")


if __name__ == "__main__":
    run_server()
