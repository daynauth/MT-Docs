import base64
import os
import json
import dataclasses
from dataclasses import make_dataclass, fields
from typing import Type, Any

from pydantic import create_model, BaseModel


def path_safe_model_name(model_name: str) -> str:
    """
    Convert model name to a path-safe format by replacing '-' and ':' with '_'.
    """
    return model_name.replace('-', '_').replace(':', '_')


def read_json(file_path: str) -> dict:
    """
    Read a JSON file and return its content as a dictionary.
    If the file does not exist, return an empty dictionary.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Returning empty dictionary.")
        return {}

    with open(file_path, 'r') as f:
        return json.load(f)


def write_json(data: dict[str, Any], file_path: str, overwrite: bool = False):
    """
    Write data to a JSON file. If the file already exists, it will either skip writing or overwrite it based on the `rewrite` flag.
    :param data:
    :param file_path:
    :param overwrite:
    :return:
    """
    if os.path.exists(file_path):
        if not overwrite:
            print(f"File {file_path} already exists. Skipping write.")
            return
        else:
            print(f"File {file_path} already exists. Overwriting.")


    with open(file_path, 'w') as f:
        f.write(json.dumps(data, indent=4))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pydantic_to_dataclass(instance: type[BaseModel]) -> Any:
    """
    Convert a Pydantic class to a dataclass.
    """
    class_name = instance.__name__

    field_list = []

    for field_name, field_info in instance.model_fields.items():
        field_type = field_info.annotation
        if field_type is None:
            field_type = str

        field_list.append((field_name, field_type))

    return make_dataclass(class_name, field_list)


def dataclass_to_pydantic_model(dataclass_cls: Type[Any]) -> Type[BaseModel]:
    """Converts a standard dataclass to a Pydantic BaseModel."""
    field_definitions = {}
    for field in fields(dataclass_cls):
        field_definitions[field.name] = (field.type, field.default if field.default is not dataclasses.MISSING else ...)

    return create_model(dataclass_cls.__name__ + "Pydantic", **field_definitions)
