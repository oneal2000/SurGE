import argparse
import os
import gdown
from evaluator import SurGEvaluator

def ensure_data_files():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    #下载 corpus.json
    corpus_path = os.path.join(data_dir, "corpus.json")
    if not os.path.exists(corpus_path):
        gdown.download(
            "https://drive.google.com/uc?id=1QAZV9nL7xhtkdiiqPpM0I8do6cF7mY3j",
            corpus_path,
            quiet=False
        )
    
    # 下载 surveys.json
    surveys_path = os.path.join(data_dir, "surveys.json")
    if not os.path.exists(surveys_path):
        gdown.download(
            "https://drive.google.com/uc?id=1-hK_RrOwvvMQ5Zk9Pnsn6c67pGokGWqp",
            surveys_path,
            quiet=False
        )

def parse_args():
    parser = argparse.ArgumentParser(description='Survey Generation Evaluator')
    parser.add_argument(
        '--passage_dir',
        type=str,
        help='Directory containing generated survey passages'
    )
    parser.add_argument(
        '--eval_list',
        nargs='+',
        default=["ALL"],
        help='Evaluation metrics to compute (space-separated)'
    )
    parser.add_argument(
        '--survey_path',
        type=str,
        default=os.path.join("data", "surveys.json"),
        help='Path to surveys.json file'
    )
    parser.add_argument(
        '--corpus_path', 
        type=str,
        default=os.path.join("data", "corpus.json"),
        help='Path to corpus.json file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="0",
        help='Device ID for computation'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default="sk-omIoPRsaYiBOkedF21B8D4Dd515d4782A69f50F679C38fF2",
        help='API key for evaluation services'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save evaluation results'
    )
    return parser.parse_args()

if __name__ == '__main__':
    ensure_data_files()
    args = parse_args()

    evaluator = SurGEvaluator(
        device=args.device,
        survey_path=args.survey_path,
        corpus_path=args.corpus_path,
        api_key=args.api_key
    )

    result = evaluator.eval_all(
        passage_dir=args.passage_dir,
        eval_list=args.eval_list,
        save_path=args.save_path
    )

    print("\nEvaluation Results:")
    print(result)