import argparse
import os
from evaluator import SurGEvaluator


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