import argparse
import os
import gdown
from evaluator import SurGEvaluator

def ensure_data_files():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    corpus_path = os.path.join(data_dir, "corpus.json")
    if not os.path.exists(corpus_path):
        gdown.download(
            "https://drive.google.com/uc?id=1QAZV9nL7xhtkdiiqPpM0I8do6cF7mY3j",
            corpus_path,
            quiet=False
        )
    
    surveys_path = os.path.join(data_dir, "surveys.json")
    if not os.path.exists(surveys_path):
        gdown.download(
            "https://drive.google.com/uc?id=1-hK_RrOwvvMQ5Zk9Pnsn6c67pGokGWqp",
            surveys_path,
            quiet=False
        )
        
if __name__ == '__main__':
    ensure_data_files()