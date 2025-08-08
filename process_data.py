import pandas as pd
import os
import json

data_dir = './data'
data_file = f"{data_dir}/test.parquet"
df = pd.read_parquet(data_file)

image_dir = f"{data_dir}/images"
annotations_dir = f"{data_dir}/annotations"

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

if not os.path.exists(annotations_dir):
    os.makedirs(annotations_dir)

for index, row in df.iterrows():
    image_path = os.path.join(image_dir, f"{index}.png")

    with open(image_path, 'wb') as f:
        f.write(row['image'])  # write image data to file

    annotation_path = os.path.join(annotations_dir, f"{index}.json")

    with open(annotation_path, 'w') as f:
        f.write(json.dumps(row['annotations'], indent=4))



print(df.head())