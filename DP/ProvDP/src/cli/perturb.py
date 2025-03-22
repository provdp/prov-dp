import argparse
import gc
import inspect
import json
from pathlib import Path
import random

from tqdm import tqdm

from src import GraphProcessor, Tree


def run_processor(args):
    input_paths = list(args.input_dir.rglob("nd*.json"))
    # Apply graph limit
    if args.num_graphs is not None:
        input_paths = random.sample(input_paths, args.num_graphs)
        args.output_dir = args.output_dir.with_stem(
            f"{args.output_dir.stem}_N={args.num_graphs}"
        )
    args.output_dir = args.output_dir.with_stem(
        f"{args.output_dir.stem}"
        f"_epsilon={args.epsilon}"
        f"_delta={args.delta}"
        f"_alpha={args.alpha}"
        f"_beta={args.beta}"
        f"_gamma={args.gamma}"
        f"_eta={args.eta}"
        f"_k={args.k}"
    )
    print(
        f"ProvDP: started run with INPUT dir: {args.input_dir} and OUTPUT dir: {args.output_dir.name}"
    )

    # Run graph processor
    graph_processor = GraphProcessor(**to_processor_args(args))
    perturbed_graphs: list[Tree] = graph_processor.perturb_graphs(input_paths)

    # Save dot files
    for graph in tqdm(perturbed_graphs, desc="Saving graphs"):
        base_file_name = f"nd_{graph.graph_id}_processletevent"
        file_path = args.output_dir / base_file_name / f"{base_file_name}.json"
        graph.write_dot(file_path)

        with open(file_path, "w") as f:
            f.write(graph.to_json())

    # Write stats to json
    with open(args.output_dir / "processor_stats.json", "w") as f:
        f.write(json.dumps(graph_processor.stats))

    # Clean up for the next run
    del graph_processor
    del perturbed_graphs
    gc.collect()


def to_processor_args(args):
    # Map args to GraphProcessor constructor
    parameters = inspect.signature(GraphProcessor.__init__).parameters
    processor_args = {}
    for arg, value in vars(args).items():
        if arg not in parameters:
            # print(f"WARNING: {arg} not in parameters")
            continue
        processor_args[arg] = value

    return processor_args

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--input_dir", type=Path, help="Path to input graph directory"
    )

    # GraphProcessor arguments
    arg_parser.add_argument(
        "-N",
        "--num_graphs",
        type=int,
        default=500,
        help="Limit the number of graphs to process",
    )
    arg_parser.add_argument(
        "-o", "--output_dir", type=Path, help="Path to output graph directory"
    )

    # Differential privacy parameters
    arg_parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=1,
        help="Differential privacy budget for pruning",
    )
    arg_parser.add_argument(
        "-d",
        "--delta",
        type=float,
        default=0.5,
        help="Allocation of privacy budget between algorithm 1 and 2",
    )

    arg_parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.25,
        help="Weight of subtree size on pruning probability",
    )
    arg_parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=0.25,
        help="Weight of subtree height on pruning probability",
    )
    arg_parser.add_argument(
        "-c",
        "--gamma",
        type=float,
        default=0.25,
        help="Weight of subtree depth on pruning probability",
    )
    arg_parser.add_argument(
        "--eta",
        type=float,
        default=0.25,
        help="Weight of node degree on pruning probability",
    )
    arg_parser.add_argument(
        "--k",
        type=float,
        default=0.10,
        help="Maximum modification per tree (percentage %)",
    )


    # Algorithm configuration
    arg_parser.add_argument(
        "-s",
        "--single_threaded",
        action="store_true",
        help="Disable multiprocessing (for debugging)",
    )

    # Checkpoint flags
    arg_parser.add_argument(
        "-p",
        "--load_perturbed_graphs",
        action="store_true",
        help="Load perturbed graphs from output directory",
    )
    return arg_parser.parse_args()



if __name__ == "__main__":
    run_processor(parse_args())
