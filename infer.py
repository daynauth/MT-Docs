from mtllm import Image

from dataclasses import dataclass
from platforms.mtllm import MTLLMPlatform
from platforms.litellm import LiteLLMPlatform
from platforms.baseplatform import BasePlatform

from tqdm import tqdm

from utils import dataclass_to_pydantic_model, write_json

import argparse
import os

from pydantic import BaseModel
@dataclass
class Invoice:
    date: str
    doc_no_receipt_no: str
    seller_address: str
    seller_gst_id: str
    seller_name: str
    seller_phone: str
    total_amount: str
    total_tax: str

def key_information_extraction(document: Image) -> Invoice:
    ...



def run_batch_inference(
        infer_platform: BasePlatform,
        start: int = 0,
        end: int = 10,
        overwrite: bool = False
) -> None:

    data_dir = './data'
    image_dir = f"{data_dir}/images"

    # Get number of image in the directory
    num_images = len([name for name in os.listdir(image_dir) if name.endswith('.png')])

    # set the end to num_images if it exceeds the number of images
    if end > num_images:
        end = num_images


    results_dir = f"./results/{infer_platform.name}/{infer_platform.dir_name}"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for index in tqdm(range(start, end), desc="Inference"):
        image_url = f"{image_dir}/{index}.png"

        # check if the image file exists
        if not os.path.exists(image_url):
            print(f"Image file {image_url} does not exist. Skipping.")
            continue

        json_file_path = f"{results_dir}/{index}.json"

        if os.path.exists(json_file_path) and not overwrite:
            continue

        invoice = infer_platform.run(image_url)
        write_json(invoice, json_file_path, overwrite = overwrite)




def inference(platform: str = "mtllm", overwrite: bool = False) -> None:
    model_name = 'ollama/gemma3:12b-it-qat'

    if platform == "mtllm":
        infer_platform = MTLLMPlatform(model=model_name, infer_function=key_information_extraction)
    elif platform == "litellm":
        system_prompt = """
        You are a highly accurate document understanding agent designed to extract structured information from scanned receipts, invoices, and sales slips.

        Your goal is to extract a fixed set of predefined fields from a given document image and return them as a single well-formed JSON object.

        Follow these rules carefully:
        1. Match Labels and Synonyms: Use exact field labels or common variations (e.g., "Tax ID", "GST No.", "TIN").
        2. Position Awareness: Use the layout of the document to infer missing labels (e.g., phone number near store name).
        3. Text Cleanup: Remove OCR noise, headers, and irrelevant content.
        4. Currency Handling: Preserve currency symbols and decimal formatting in monetary values.
        5. Missing or Unreadable Fields: If a field is not present or unreadable, return its value as empty string.
        6. Field Consistency: Always return the same 8 fields, in the exact order shown below.

        Output format must strictly match this schema:
        {
          "date": "DD/MM/YYYY or similar format",
          "doc_no_receipt_no": "...",
          "seller_name": "...",
          "seller_address": "...",
          "seller_phone": "...",
          "seller_gst_id": "...",
          "total_tax": "...",
          "total_amount": "..."
        }
        """


        human_prompt = """
            Extract the following fields from the given document image:
            - date
            - doc_no_receipt_no
            - seller_name
            - seller_address
            - seller_phone
            - seller_gst_id
            - total_tax
            - total_amount

            Return a single JSON object that includes all fields, with empty string where necessary.
            """

        # todo: this is too icky, need to fix
        infer_platform = LiteLLMPlatform(model_name, human_prompt, system_prompt, dataclass_to_pydantic_model(Invoice))
    else:
        raise ValueError(f"Unsupported platform: {platform}")


    run_batch_inference(infer_platform, 0, 10, overwrite=overwrite)

def main(platform: str, overwrite: bool) -> None:
    # inference(platform="mtllm")
    inference(platform=platform, overwrite = overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on document images.")
    parser.add_argument('-p', '--platform', type=str, choices=['mtllm', 'litellm'], default='mtllm',
                        help="Platform to use for inference (default: mtllm)")
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)


    args = parser.parse_args()

    main(args.platform, args.overwrite)