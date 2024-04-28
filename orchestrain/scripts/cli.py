from __future__ import annotations

import sys

import fire
import scripts.train as train
from scripts.converter import model_jit_converter
from scripts.render import renderer_fn

from orchestrain.version import VERSION


def show_version():
    """prints the version"""
    print(VERSION)


app = {"version": show_version, "render": renderer_fn, "convert": model_jit_converter, **train.fire_train_type_selector}


def main(args=None):
    fire.Fire(app)


if __name__ == "__main__":
    sys.exit(main())
