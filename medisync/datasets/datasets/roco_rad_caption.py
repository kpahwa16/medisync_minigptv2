import os
import json
import pickle
import random
import time
import itertools

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds

from medisync.datasets.datasets.base_dataset import BaseDataset
from medisync.datasets.datasets.caption_datasets import CaptionDataset


import os
import json
import random
import logging
from PIL import Image
from torch.utils.data import Dataset

# Set up logging 
logging.basicConfig(level=logging.INFO)
class ROCORADCapDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g., coco/images/)
        ann_path (string): Path to the annotation JSON file
        """
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.instruction_pool = [
            'Briefly describe this image.',
            'Provide a concise depiction of this image.',
            'Present a short description of this image.',
            'Summarize this image in a few words.',
            'A short image caption:',
            'A short image description:',
            'A photo of ',
            'An image that shows ',
            'Write a short description for the image.',
            'Write a description for the photo.',
            'Provide a description of what is presented in the photo.',
            'Briefly describe the content of the image.',
            'Can you briefly explain what you see in the image?',
            'Could you use a few words to describe what you perceive in the photo?',
            'Please provide a short depiction of the picture.',
            'Using language, provide a short account of the image.',
            'Use a few words to illustrate what is happening in the picture.',
        ]


        with open(ann_path, 'r') as f:
            ann = json.load(f)
            
            if not isinstance(ann, list):
                raise ValueError("Expected the JSON file to contain a list of annotations.")
            self.valid_ann = [a for a in ann if os.path.exists(os.path.join(vis_root, a["image"]))]
        logging.info(f"Loaded {len(self.valid_ann)} valid annotations")


    def __len__(self):
        return len(self.valid_ann)

    def __getitem__(self, index):
        info = self.valid_ann[index]
        image_file = info["image"]
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        caption = info["caption"]
        caption = self.text_processor(caption)
        instruction = f"<Img><ImageHere></Img> [caption] {random.choice(self.instruction_pool)}"
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
        }
