import argparse
import ast
import logging
from typing import List

logger = logging.getLogger(__name__)


def parse_ks_arg(value: str) -> List[int]:
    try:
        ks = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--Ks must be a Python-style list such as '[5, 10, 20]'"
        ) from exc

    if not isinstance(ks, (list, tuple)) or not ks:
        raise argparse.ArgumentTypeError("--Ks must be a non-empty list of integers")

    if any(not isinstance(k, int) or k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("--Ks must contain positive integers only")

    return list(ks)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument(
            "--checkpoint_dir", type=str, default="runlog/", help="logs are saved here"
        )
        self.parser.add_argument(
            "--logging_dir", type=str, default="runlog/", help="logging output path."
        )
        self.parser.add_argument("--best_model_path", type=str, help="load best model.")
        self.parser.add_argument(
            "--all_models_path", type=str, help="load all saved model."
        )
        self.parser.add_argument(
            "--data_dir", nargs="?", default="data/", help="Input data path."
        )
        self.parser.add_argument(
            "--data_name",
            nargs="?",
            default="beauty",
            help="Choose a dataset from {yelp , beauty,sports,toys}",
        )
        self.parser.add_argument(
            "--experiment_name",
            nargs="?",
            default="name",
            help="exp name path. name or address."
            "This parameter is used only for the convenience of differentiating experiments",
        )
        self.parser.add_argument("--seed", type=int, default=42, help="random seed ")
        self.parser.add_argument(
            "--item_size",
            type=int,
            default=32,
            help="maximum number of tokens in item text",
        )
        self.parser.add_argument(
            "--seq_size",
            type=int,
            default=256,
            help="maximum number of tokens in item text",
        )
        self.parser.add_argument(
            "--split_num", type=int, default=243, help="item num of seq "
        )
        self.parser.add_argument(
            "--num_passage", type=int, default=2, help="item num of seq "
        )
        self.parser.add_argument(
            "--eval_batch_size", type=int, default=16, help="batch size."
        )
        self.parser.add_argument(
            "--Ks",
            type=parse_ks_arg,
            default=[5, 10, 15, 20],
            help="Calculate metric@K when evaluating.",
        )
        self.parser.add_argument(
            "--stopping_step", type=int, default=0, help="early stop"
        )

    def parse(self):
        opt = self.parser.parse_args()
        return opt
