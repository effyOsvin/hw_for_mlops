import torch


def save_model(model, model_parameters, save_path, save_name):
    model_dict = model.state_dict()
    tmp_save = [model_dict, model_parameters]
    torch.save(tmp_save, save_path + save_name)
