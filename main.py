import os
import csv
import uuid
import logging
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch

from dataloader import get_dataloader
from models.load_encoder import MODELS, load_encoder
from palate import compute_palate
from representations import get_representations

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger.info("Starting main.py")

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model",
    type=str,
    default="dinov2",
    choices=MODELS.keys(),
    help="Model to use for generating feature representations.",
)

parser.add_argument(
    "-bs", "--batch_size", type=int, default=50, help="Batch size to use"
)

parser.add_argument(
    "--num-workers",
    type=int,
    help="Number of processes to use for data loading. "
    "Defaults to `min(8, num_cpus)`",
)

parser.add_argument(
    "--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu"
)


parser.add_argument(
    "--nsample",
    type=int,
    default=10000,
    help="Maximum number of images to use for calculation",
)

parser.add_argument(
    "path",
    type=str,
    nargs="+",
    help="Paths to the images. The order is train, test, gen_1, gen_2, ..., gen_n ",
)


parser.add_argument(
    "--output_dir",
    type=str,
    default="output/",
    help="Directory for saving outputs: metrics_summary.csv, metrics_summary.txt and arguments.txt",
)

parser.add_argument(
    "--save", action="store_true", help="Save representations to repr dir"
)

parser.add_argument(
    "--load",
    action="store_true",
    help="Load representations and statistics from previous runs if possible",
)

parser.add_argument(
    "--no-load",
    action="store_false",
    dest="load",
    help="Do not load representations and statistics from previous runs.",
)
parser.set_defaults(load=True)

parser.add_argument("--seed", type=int, default=13579, help="Random seed")

parser.add_argument(
    "--clean_resize", action="store_true", help="Use clean resizing (from pillow)"
)

parser.add_argument(
    "--depth",
    type=int,
    default=0,
    help="Negative depth for internal layers, positive 1 for after projection head.",
)


parser.add_argument(
    "--repr_dir",
    type=str,
    default="./saved_representations",
    help="Dir to store saved representations.",
)


def get_device_and_num_workers(device, num_workers):
    if device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(device)

    if num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = num_workers

    logger.info(f"Device: {device}, num_workers: {num_workers}")

    return device, num_workers


def get_dataloader_from_path(
    path, model_transform, num_workers, args, sample_w_replacement=False
):
    logger.info(f"Initializing dataloader for path: {path}")
    dataloader = get_dataloader(
        path,
        args.nsample,
        args.batch_size,
        num_workers,
        seed=args.seed,
        sample_w_replacement=sample_w_replacement,
        transform=lambda x: model_transform(x),
    )
    return dataloader


def create_unique_output_name():
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = uuid.uuid4()
    logger.info(f"Generated unique output name: {unique_str}")
    return str(unique_str)[:8]


def write_to_txt(scores, output_dir, model, train_path, test_path, gen_path, nsample):
    out_file = "metrics_summary.txt"
    out_path = os.path.join(output_dir, out_file)

    with open(out_path, "a") as f:
        f.write(f"Model: {model}\n")
        f.write(
            f"Train: {train_path}, Test: {test_path}, Gen: {gen_path}, Nsample: {nsample}\n"
        )
        for key, value in scores.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "=" * 50 + "\n\n")


def write_to_csv(scores, output_dir, train_name, test_name, gen_name, nsample):
    csv_file = os.path.join(output_dir, "metrics_summary.csv")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Train", "Test", "Gen", "Nsample"] + list(scores.keys())
            writer.writerow(header)
        row = [train_name, test_name, gen_name, nsample] + list(scores.values())
        writer.writerow(row)


def get_last_x_dirs(path, x=2):
    parts = pathlib.Path(path).parts
    x = min(x, len(parts))
    return "_".join(parts[-x:])


def save_score(scores, output_dir, model, train_path, test_path, gen_path, nsample):

    train_name = get_last_x_dirs(train_path)
    test_name = get_last_x_dirs(test_path)
    gen_name = get_last_x_dirs(gen_path)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_to_txt(scores, output_dir, model, train_path, test_path, gen_path, nsample)
    write_to_csv(scores, output_dir, train_name, test_name, gen_name, nsample)

    logger.info(f"Saved scores to {output_dir}")


def get_model(args, device):
    return load_encoder(
        args.model,
        device,
        ckpt=None,
        arch=None,
        clean_resize=args.clean_resize,
        sinception=True if args.model == "sinception" else False,
        depth=args.depth,
    )


def compute_representations(path, model, num_workers, device, args):
    """
    Compute or load representations for the given path.

    Args:
        path (str): Path to the data.
        model: The model used to compute representations.
        num_workers (int): Number of workers for the dataloader.
        device: The device to use for computation (e.g., "cuda" or "cpu").
        args: Command-line arguments or configuration object.

    Returns:
        np.ndarray: Computed or loaded representations.
    """

    if args.load:
        loaded_reps = load_reps_from_path(args.repr_dir, path, model, args.nsample)
        if loaded_reps is not None:
            return loaded_reps

    logger.info("Load path doesn't exist")
    dataloader = get_dataloader_from_path(path, model.transform, num_workers, args)

    logger.info(f"Computing representations for path: {path}")
    representations = get_representations(model, dataloader, device, normalized=False)

    if args.save:
        save_outputs(
            args.repr_dir, path, representations, model, dataloader, args.nsample
        )

    return representations

    """Save representations and other info to disk at file_path"""


def save_outputs(output_dir, path, reps, model, dataloader, nsample):
    # Create a unique file path for saving
    out_path = get_path(output_dir, path, model, nsample)

    # Ensure the output directory exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare hyperparameters for saving
    hyperparams = vars(dataloader).copy()  # Remove keys that can't be pickled
    hyperparams.pop("transform", None)
    hyperparams.pop("data_loader", None)
    hyperparams.pop("data_set", None)

    logger.info(f"Saving representations to: {out_path}")
    np.savez(out_path, model=model, reps=reps, hparams=hyperparams)


def load_reps_from_path(saved_dir, path, model, nsample):
    """
    Load representations from a saved .npz file if it exists.

    Args:
        saved_dir (str): Directory where the representations are saved.
        model: The model used to generate the representations.
        dataloader: The dataloader used to generate the representations.
        nsample (int): Number of samples used to generate the representations.

    Returns:
        np.ndarray: Loaded representations if the file exists, otherwise None.
    """
    # Generate the file path
    load_path = get_path(saved_dir, path, model, nsample)
    logger.info(f"Checking if load path exists: {load_path}")

    if os.path.exists(f"{load_path}"):
        saved_file = np.load(f"{load_path}")
        reps = saved_file["reps"]
        print(f"Loaded representations from {load_path}")
        return reps
    else:
        return None


def get_path(output_dir, path, model, nsample):
    """Generate a unique file path for saving representations"""
    # Example: Create a unique name based on model, checkpoint, and dataloader
    model_name = model.__class__.__name__

    dataset_name = get_last_x_dirs(path)

    return os.path.join(output_dir, f"{model_name}_{dataset_name}_{str(nsample)}.npz")


def write_arguments(args, output_dir, filename="arguments.txt"):
    """
    Writes all arguments from the `args` object to a file, one argument per line.

    Args:
        args: The argparse.Namespace object containing the arguments.
        output_dir: The directory where the file will be saved.
        filename: The name of the file to write the arguments to.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w") as f:
        for arg in vars(args):
            value = getattr(args, arg)
            f.write(f"{arg}: {value}\n")


def main():
    args = parser.parse_args()
    device, num_workers = get_device_and_num_workers(args.device, args.num_workers)
    if len(args.path) < 3:
        logger.error(
            "At least three paths are required: train, test, and one or more generated."
        )
        return

    train_path = args.path[0]
    test_path = args.path[1]
    gen_paths = args.path[2:]

    logger.info(f"Loading model: {args.model}")
    model = get_model(args, device)
    logger.info(f"Loaded model: {args.model}")
    train_representations = compute_representations(
        train_path, model, num_workers, device, args
    )
    logger.info("Finished loading/computing train representations")

    test_representations = compute_representations(
        test_path, model, num_workers, device, args
    )
    logger.info("Finished loading/computing test representations")
    scores = {}
    unique_name = create_unique_output_name()
    output_dir = os.path.join(args.output_dir, unique_name)
    write_arguments(args, output_dir)
    logger.info(f"Enumerating paths to generated samples: {gen_paths}")
    for gen_path in gen_paths:

        gen_representations = compute_representations(
            gen_path, model, num_workers, device, args
        )

        m_palate, palate = compute_palate(
            train_representations=train_representations,
            test_representations=test_representations,
            gen_representations=gen_representations,
        )

        scores["m_palate"] = m_palate
        scores["palate"] = palate

        save_score(
            scores,
            output_dir,
            model,
            train_path,
            test_path,
            gen_path,
            args.nsample,
        )


if __name__ == "__main__":
    main()
