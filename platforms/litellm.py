from utils import encode_image
from platforms.baseplatform import BasePlatform

from litellm import completion
from pydantic import BaseModel
import json

from typing import Type

class LiteLLMPlatform(BasePlatform):
    def __init__(
            self,
            model: str,
            user_prompt_template: str,
            system_prompt: str | None = None,
            schema: Type[BaseModel] | None = None
    ):
        super().__init__(model, "litellm")
        self.schema = schema
        self.messages: list = []

        if system_prompt:
            system_message = {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }
            self.messages.append(system_message)

        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt_template
                },
                {
                    "type": "image_url",
                    "image_url": "",
                }
            ]
        }
        self.messages.append(user_message)
        print(self.messages)





    def run(self, image_url: str):
        image_base64 = encode_image(image_url)

        self.messages[-1]["content"][1]["image_url"] = f"data:image/jpeg;base64,{image_base64}"

        response = completion(
            model= self.model,
            api_base="http://localhost:11434",
            response_format=self.schema,
            temperature=0,
            messages=self.messages,
        )

        output = response.choices[0].message.content
        print(output)
        return json.loads(output)