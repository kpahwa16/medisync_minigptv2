import os
import json
import random
import logging
from PIL import Image
from torch.utils.data import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)

class MIMICCapDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.instruction_pool = [
            'Briefly describe this image.', 'Provide a concise depiction of this image.', 'Present a short description of this image.', 
            'Summarize this image in a few words.', 'A short image caption:', 'A short image description:', 'A photo of ', 
            'An image that shows ', 'Write a short description for the image.', 'Write a description for the photo.',
            'Provide a description of what is presented in the photo.', 'Briefly describe the content of the image.',
            'Can you briefly explain what you see in the image?', 'Could you use a few words to describe what you perceive in the photo?',
            'Please provide a short depiction of the picture.', 'Using language, provide a short account of the image.',
            'Use a few words to illustrate what is happening in the picture.',
        ]

        if not isinstance(ann_path, str):
            raise TypeError(f"ann_path must be a string, got {type(ann_path)}")

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

        if not isinstance(self.ann, list):
            raise ValueError("Expected the JSON file to contain a list of annotations.")

        self.valid_ann = [a for a in self.ann if os.path.exists(os.path.join(self.vis_root, f"{a['image_id']}.jpg"))]
        logging.info(f"Loaded {len(self.valid_ann)} valid annotations out of {len(self.ann)}")

    def __len__(self):
        return len(self.valid_ann)

    def __getitem__(self, index):
        info = self.valid_ann[index]
        image_file = f"{info['image_id']}.jpg"
        image_path = os.path.join(self.vis_root, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
            return None  # or use a placeholder image
        
        caption = info["caption"]
        caption = self.text_processor(caption)
        instruction = f"<Img><ImageHere></Img> [caption] {random.choice(self.instruction_pool)}"
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
        }
