# mlops

## Task

Решаем задачу классификации датасета MNIST .

## Usage

### Setup

```
poetry install
pre-commit install
pre-commit run -a
```

### Run experiments

Для запуска сервера mlflow:

```
 mlflow ui -p 8888
```

Обучение модели (логи можно посмотреть в mlflow):

```
python hw_for_mlops/train.py
```

Инференс модели:

```
python hw_for_mlops/infer.py
```

Инференс модели, используя onnx Запуск сервера:

```
dvc pull ./bin/onnx_model.dvc
mlflow models serve -p 8880 -m ./bin/onnx_model --env-manager=local
```

Запуск инференса

```
python hw_for_mlops/run_server.py
```

Предсказания хранятся в predictions/test_result_onx.json
