from mtllm import Model, Image, by
from platforms.baseplatform import BasePlatform

from collections.abc import Callable
from dataclasses import asdict

class MTLLMPlatform(BasePlatform):
    def __init__(self, model: str, infer_function: Callable):
        super().__init__(model, "mtllm")

        self.llm = Model(model_name=self.model, temperature=0, api_base="http://localhost:11434")
        self.by = by(self.llm)
        self.infer_function = self.by(infer_function)

    def run(self, image_url: str) -> dict:
        document = Image(image_url)
        return asdict(self.infer_function(document))



