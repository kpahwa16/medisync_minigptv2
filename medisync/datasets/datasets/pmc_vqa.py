import os
import json
import random
from PIL import Image
from medisync.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from collections import OrderedDict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict({
            "file": ann["image"],
            "question": ann["question"],
            "question_id": ann["question_id"],
            "answers": ann["answer"],
            "image": sample["image"],
        })

class PMCVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        # Handle ann_paths as a list or single path
        if isinstance(ann_paths, list):
            ann_path = ann_paths[0]  # Assuming we want the first file
        else:
            ann_path = ann_paths

        with open(ann_path, 'r') as f:
            annotations = json.load(f)

        # Validate and filter annotations with correct image paths
        self.annotation = [
            ann for ann in annotations
            if self._is_valid_image(os.path.join(self.vis_root, ann["image"]))
        ]

    def _is_valid_image(self, image_path):
        if os.path.exists(image_path) and os.path.isfile(image_path):
            return True
        else:
            logging.warning(f"Invalid image path encountered: {image_path}")
            return False

    def get_data(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]
        answer = ann["answer"]
        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }

class PMCVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.instruction_pool = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]
        self.vis_root = vis_root

        # Handle ann_paths consistently as done in PMCVQADataset
        if isinstance(ann_paths, list):
            ann_path = ann_paths[0]  # Assuming we want the first file
        else:
            ann_path = ann_paths

        self.annotation = json.load(open(ann_path))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        
        return {
            "image": image,
            'image_path': image_path,
            "question": question,
            "question_id": ann["question_id"],
            "instruction_input": instruction,
            "instance_id": ann.get("instance_id", index),  # Safe access to possibly non-existent key
        }
