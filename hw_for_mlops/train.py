import hydra
import lightning.pytorch as pl
import mlflow
import torch
import torch.utils.data
from models.model import ConvLinear
from omegaconf import DictConfig
from tools.data import MyDataModule
from tools.save_model import convert_to_onnx, save_model


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")
    dm = MyDataModule(cfg=cfg.train)
    model = ConvLinear(cfg.model.dropout, cfg.model.out_dim)
    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.logger.experiment_name,
            tracking_uri=cfg.logger.tracking_uri,
            artifact_location="conf/config.yaml",
        )
    ]
    loggers[0].log_hyperparams(cfg)
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(),
    ]

    trainer = pl.Trainer(
        log_every_n_steps=cfg.train.log_every_n_steps,
        max_epochs=cfg.train.num_epoch,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)

    save_model(
        model,
        cfg.model,
        save_path=cfg.model.save_path,
        save_name=cfg.model.save_name,
    )

    mlflow.set_tracking_uri(cfg.logger.tracking_uri)
    mlflow.set_experiment(cfg.logger.experiment_name)
    convert_to_onnx(model=model, conf=cfg.model.model_onnx)


if __name__ == "__main__":
    train()
