"""CLI interface for acl2025_stricta project.

"""
import argparse
import os

from .eval import predict_itemwise, store_predictions
from .models import LlamaWorkflow
from .models import MistralWorkflow
from .models import OpenAiWorkflow


def validate_dataset_path(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")

    subdirs = ['main_study', 'student_seminar']
    for subdir in subdirs:
        if not os.path.isdir(os.path.join(path, subdir)):
            raise argparse.ArgumentTypeError(f"'{path}' must contain subdirectories: {', '.join(subdirs)}.")

    return [os.path.join(path, subdir) for subdir in subdirs]


def parse_args():
    parser = argparse.ArgumentParser(description="CLI interface for acl2025_stricta project.")
    parser.add_argument(
        '--dataset-path',
        required=True,
        help="Path to the dataset directory containing 'main_study' and 'student_seminar' subdirectories."
    )
    parser.add_argument(
        '--experiment',
        choices=['predict', 'eval'],
        required=True,
        help="Mandatory choice parameter: 'predict' or 'eval'."
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        choices=["mixtral", "gpt4", "llama3"],
        help="Name of the model to be used."
    )

    return parser.parse_args()


def load_model(model_name, workflow):
    if model_name == "llama3":
        local_model_path = os.environ.get("MODEL_PATH")
        model_subpath = os.environ.get("MODEL_SUBPATH")
        model = LlamaWorkflow(workflow,
                              "llama-3",
                              model_path=local_model_path,
                              model_subpath=model_subpath,
                              compress_paper=True)  # discards references of paper to reduce context usage

        with open("./prompts/llama3/extract.txt", "r") as f:
            exprompt = f.read().strip()
        with open("./prompts/llama3/infer.txt", "r") as f:
            infprompt = f.read().strip()

        model.set_instructions(exprompt, "extract")
        model.set_instructions(infprompt, "infer")
    elif model_name == "mixtral":
        assert torch.cuda.is_available(), "This model should be run only on GPU."

        local_model_path = os.environ.get("MODEL_PATH")

        model = MistralWorkflow(workflow,
                                "mistralai--Mixtral-8x7B-Instruct-v0.1",
                                model_path=local_model_path,
                                model_subpath=None,
                                compress_paper=True)  # discarding references
        with open("./prompts/mixtral/extract.txt", "r") as f:
            exprompt = f.read().strip()
        with open("./prompts/mixtral/infer.txt", "r") as f:
            infprompt = f.read().strip()

        model.set_instructions(exprompt, "extract")
        model.set_instructions(infprompt, "infer")
    elif model_name == "gpt4":
        model = OpenAiWorkflow(workflow, "gpt-4")

        with open("./prompts/gpt4_mm/extract.txt", "r") as f:
            exprompt = f.read().strip()
        with open("./prompts/gpt4_mm/infer.txt", "r") as f:
            infprompt = f.read().strip()

        model.set_instructions(exprompt, "extract")
        model.set_instructions(infprompt, "infer")
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are 'llama3', 'mixtral', and 'gpt4o'.")

    return model


def main():  # pragma: no cover
    args = parse_args()

    print(f"Dataset Path: {args.dataset_path}")
    print(f"Mode: {args.experiment}")
    print(f"Model Name: {args.model_name}")

    datasets = validate_dataset_path(args.dataset_path)

    os.mkdirs("./results", exist_ok=True)

    if args.experiment == "predict":
        workflow, papers, annotations = load_itemwise_dataset(datasets)

        # load model
        model = load_model(args.model_name, workflow)

        predictions = predict_itemwise(
            workflow=workflow,
            papers=papers,
            itemwise_answers=annotations,
            model=model,
            _cache_dir="./results"
        )
        store_predictions(predictions, f"./results/predictions_{args.model_name}.csv")
    elif args.experiment == "eval":
        assert os.exists(f"./results/predictions_{args.model_name}.csv")

        os.mkdirs(f"./results/eval_{args.model_name}", exist_ok=True)
        eval_io(
            datasets,
            [f"./results/predictions_{args.model_name}.csv"],
            metrics = ["bertf1", "summac", "true_score"],
            output_path=f"./results/eval_{args.model_name}"
        )

    print("Finished")