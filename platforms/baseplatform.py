from typing import Callable

from abc import ABC, abstractmethod
from utils import path_safe_model_name


def generate_safe_model_name(model: str) -> str:
    """
    Generate a path-safe model name by replacing '-' and ':' with '_'.
    """
    model_name = model.split('/')[-1]

    return path_safe_model_name(model_name)


class BasePlatform(ABC):

    def __init__(self, model: str, name: str):
        self.model = model
        self.dir_name = generate_safe_model_name(model)
        self.name = name

    def run(self, image_url: str) -> dict:
        """
        Run the inference on the given image URL.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

