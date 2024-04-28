from __future__ import annotations

import fire
import numpy as np
import torch

import orchestrain.utils as utils


class ModelSpeed:
    """Measures the forward pass speed of a PyTorch model.

    Parameters
    ----------
        device: str
            Device to run the model on ("cpu" or "cuda").
        input_shape: list
            Shape of the input tensor.
        force_data: bool, optional
            Force loading data flag. Defaults to False.
        **kwargs:
            Additional keyword arguments including "config" and "weights".

    """

    def __init__(self, device, input_shape: list, force_data=True, **kwargs):
        self.device = device
        if self.device.lower() == "gpu":
            self.device = "cuda"
        self.config = kwargs["config"]
        self.log_gpu_time = self.device.startswith("cuda")
        weights = kwargs.get("weights", "")

        if weights:
            self.model, self.config["model"] = utils.load_model(weights, self.config.get("model", {}), force_config=force_data)
        else:
            self.model, self.config["model"] = utils.generate_model(self.config.get("model", {}))

        self.input_shape = input_shape
        self.model = self.model.to(self.device)
        self.model.eval()

    def measure_forward_speed(self, num_iterations):
        input_tensor = torch.rand(self.input_shape).to(self.device)

        self.cpu_times = np.empty(0)
        self.gpu_times = np.empty(0)

        for i in range(num_iterations):
            if i < 5:
                _ = self.model(input_tensor)
            else:
                with utils.Profiler.profile_time(self.log_time, "forward_pass", log_gpu_time=self.log_gpu_time):
                    _ = self.model(input_tensor)

        self.__mean_ort()

    def log_time(self, log_name, time_value):
        """Logs the time taken for a specific operation.

        Parameters
        ----------
            log_name: str
                Name of the operation.
            time_value: float
                Time taken for the operation.
        """
        if "cpu" in log_name:
            self.cpu_times = np.append(self.cpu_times, time_value)

        if "gpu" in log_name:
            self.gpu_times = np.append(self.gpu_times, time_value)

    def __mean_ort(self):
        print(f"profiler/forward_pass_cpu: {np.mean(self.cpu_times)} ms")
        if self.log_gpu_time:
            print(f"profiler/forward_pass_gpu: {np.mean(self.gpu_times)} ms")


def speed_test(device: str = "", data: str = "", weights: str = "", input_shape: list = [], num_iterations: int = 100):
    config = utils.read_yaml(data) if data else None
    force_data = config is not None

    if not (data or weights):
        raise RuntimeError("Please provide values for 'weights' and/or 'data'. Both cannot be empty.")

    if not device:
        raise RuntimeError("Please provide value for 'device'. Available options gpu, cpu, and cuda:x.")

    if not input_shape:
        raise RuntimeError("Please provide value for 'input_shape'.")

    model_speed = ModelSpeed(device=device, input_shape=input_shape, config=config, weights=weights, force_data=force_data)
    model_speed.measure_forward_speed(num_iterations)


if __name__ == "__main__":
    fire.Fire(speed_test)
