import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


from medisync.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData
from medisync.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from medisync.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from medisync.common.eval_utils import prepare_texts, init_model, eval_parser
from medisync.conversation.conversation import CONV_VISION_minigptv2
from medisync.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='slake_vqa', help="dataset to evaluate")
args = parser.parse_args()
cfg = Config(args)



model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path


if 'slake_vqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["slake_vqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["slake_vqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["slake_vqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["slake_vqa"]["max_new_tokens"]
    

    evaluation_annntation_path = os.path.join(eval_file_path, "slake_vqa_test_split.json")
    with open(evaluation_annntation_path) as f:
        ok_vqa_test_split = json.load(f)

    data = SLAKEVQAEvalDataset(ok_vqa_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    medisynq_predict = []

    for images, questions, question_ids, img_ids in eval_dataloader:
        texts = prepare_texts(questions, conv_temp) 
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, question_id, question, img_id in zip(answers, question_ids, questions, img_ids):
            result = dict()
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question_id'] = int(question_id)
            medisync_predict.append(result)

    file_save_path= os.path.join(save_path,"slake_vqa.json")
    with open(file_save_path,'w') as f:
        json.dump(medisync_predict, f)

    annFile = os.path.join(eval_file_path,"slake_vqa_test_split.json")
    quesFile = os.path.join(eval_file_path,"slake_vqa_questions_test_split.json" )

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(file_save_path, quesFile)

    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    print ("Overall SLAKE VQA accuracy is: %.02f\n" %(vqaEval.accuracy['overall']), flush=True)



if 'pmc_vqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["pmc_vqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["pmc_vqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["pmc_vqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["pmc_vqa"]["max_new_tokens"]
    

    evaluation_annntation_path = os.path.join(eval_file_path, "pmc_vqa_test_split.json")
    with open(evaluation_annntation_path) as f:
        slake_vqa_test_split = json.load(f)

    data = PMCVQAEvalDataset(ok_vqa_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    medisynq_predict = []

    for images, questions, question_ids, img_ids in eval_dataloader:
        texts = prepare_texts(questions, conv_temp) 
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, question_id, question, img_id in zip(answers, question_ids, questions, img_ids):
            result = dict()
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question_id'] = int(question_id)
            medisync_predict.append(result)

    file_save_path= os.path.join(save_path,"pmc_vqa.json")
    with open(file_save_path,'w') as f:
        json.dump(medisync_predict, f)

    annFile = os.path.join(eval_file_path,"pmc_vqa_test_split.json")
    quesFile = os.path.join(eval_file_path,"pmc_vqa_questions_test_split.json" )

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(file_save_path, quesFile)

    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    print ("Overall PMC VQA accuracy is: %.02f\n" %(vqaEval.accuracy['overall']), flush=True)


if 'vqarad' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["vqarad"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["vqarad"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vqarad"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vqarad"]["max_new_tokens"]
    

    evaluation_annntation_path = os.path.join(eval_file_path, "vqa_rad_test_split.json")
    with open(evaluation_annntation_path) as f:
        vqa_rad_test_split = json.load(f)

    data = PMCVQAEvalDataset(vqa_rad_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    medisynq_predict = []

    for images, questions, question_ids, img_ids in eval_dataloader:
        texts = prepare_texts(questions, conv_temp) 
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, question_id, question, img_id in zip(answers, question_ids, questions, img_ids):
            result = dict()
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question_id'] = int(question_id)
            medisync_predict.append(result)

    file_save_path= os.path.join(save_path,"vqa_rad.json")
    with open(file_save_path,'w') as f:
        json.dump(medisync_predict, f)

    annFile = os.path.join(eval_file_path,"vqa_rad_test_split.json")
    quesFile = os.path.join(eval_file_path,"vqa_rad_questions_test_split.json" )

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(file_save_path, quesFile)

    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    print ("Overall VQA-RAD accuracy is: %.02f\n" %(vqaEval.accuracy['overall']), flush=True)
