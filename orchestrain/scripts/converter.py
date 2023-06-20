from __future__ import annotations

import os
from pathlib import Path

import fire
import torch

import orchestrain.utils as utils


def convert_for_double_input_trainer(model, out_path, image_sizes):
    prev = torch.rand(1, 1, image_sizes[0], image_sizes[1])
    next = torch.rand(1, 1, image_sizes[0], image_sizes[1])

    traced_script_module = torch.jit.trace(model, (prev, next))
    traced_script_module.save(out_path)

    print(utils.TerminalColors.GREEN + f"Model saved to {out_path}" + utils.TerminalColors.END)


def model_jit_converter(models: list[str], data: str = "", out: str = "", img_size: tuple[int, int] = (1920, 1072)):
    """Converts the torch models to torch script

    Parameters
    ----------
    model : list[str]
        model list, model can be wandb path or real path in disc
    data : str, optional
        config yaml path. If model has class informations no needed to this, by default ""
    out : str, optional
        output directory path, by default ""
    img_size : list[int, int], optional
        sample image size, by default (1920, 1072)

    """
    config = utils.read_yaml(data) if data else {"model": None}

    models_list = list(models) if type(models) == str else models

    for model_path in models_list:
        model, config["model"] = utils.load_model(model_path, model_config=config["model"], force_config=False)
        model = model.to("cpu")
        model.eval()

        out_dir = os.path.join(out, model_path.split(".")[0]) + ".pt"
        out_dir = out_dir.replace(":", "-")
        os.makedirs(Path(out_dir).parent.absolute(), exist_ok=True)

        if os.path.exists(out_dir):
            print(utils.TerminalColors.RED + f"File exists: {out_dir}! Weight will be updated." + utils.TerminalColors.END)

        try:
            convert_for_double_input_trainer(model, out_dir, img_size)
        except Exception as e:
            print(utils.TerminalColors.RED + f"Error converting model: {model_path}. Error: {str(e)}" + utils.TerminalColors.END)


if __name__ == "__main__":
    fire.Fire(model_jit_converter)
