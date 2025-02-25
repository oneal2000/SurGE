import json
import argparse
import pandas as pd
import torch
import time
import re
from markdownParser import *
from rouge_score import rouge_scorer
from statistics import mean

import sacrebleu


def calculate_average_rouge_bleu(A, B):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    
    for a in A:
        max_rouge1, max_rouge2, max_rougeL = 0, 0, 0
        for b in B:

            scores = scorer.score(a, b)
            max_rouge1 = max(max_rouge1, scores['rouge1'].fmeasure)
            max_rouge2 = max(max_rouge2, scores['rouge2'].fmeasure)
            max_rougeL = max(max_rougeL, scores['rougeL'].fmeasure)
        
        rouge1_scores.append(max_rouge1)
        rouge2_scores.append(max_rouge2)
        rougeL_scores.append(max_rougeL)
        bleu_scores.append(sacrebleu.sentence_bleu(a, B).score)
        #print(sacrebleu.sentence_bleu(a, B).score)
    
    avg_rouge1 = mean(rouge1_scores)
    avg_rouge2 = mean(rouge2_scores)
    avg_rougeL = mean(rougeL_scores)
    #print(bleu_scores)
    avg_BLEU = mean(bleu_scores)
    
    
    return avg_rouge1, avg_rouge2, avg_rougeL, avg_BLEU


def get_content_list(psg_node:MarkdownNode):
    res = []
    text = "\n".join(psg_node.content)
    if len(text) >= 100:
        res.append(text)
    for child in psg_node.children:
        tmp_res = get_content_list(child)
        res.extend(tmp_res)
        
    return res

def eval_rougeBleu(target_survey,psg_node: MarkdownNode):
    
    target_content = []

    for section in target_survey['structure']:
        if len(section['content']) < 100:
            continue
        target_content.append(section['content'])

        
    gen_content = get_content_list(psg_node)
    #print(gen_content)
    return calculate_average_rouge_bleu(gen_content,target_content)
    
