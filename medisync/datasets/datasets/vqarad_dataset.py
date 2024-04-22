import os
import json
import random
from PIL import Image
from medisync.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from collections import OrderedDict

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict({
            "file": ann["image"],
            "question": ann["question"],
            "question_id": ann["question_id"],
            "answers": ann["answer"],  # Assuming 'answer' is a string; remove the join if so
            "image": sample["image"],
        })

class VQARADDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        # Filter existing annotations based on image file presence
        self.annotation = [ann for ann in self.annotation if os.path.exists(os.path.join(self.vis_root, ann["image"].split('/')[-1]))]

    def get_data(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]
        answer = ann["answer"]  # Direct use of the single answer provided

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
        if isinstance(data['answer'], int):
            data['answer'] = str(data['answer'])  # Convert to string if necessary
        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }

