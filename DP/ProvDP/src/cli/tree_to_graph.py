import argparse
from pathlib import Path

from tqdm import tqdm

from src.algorithm.wrappers.graph import Graph
from src.algorithm.wrappers.tree import Tree


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    input_paths = list(input_dir.rglob("nd*.json"))
    print(f"Preprocessing {input_dir}")

    # Run graph processor
    for input_path in tqdm(input_paths, desc="Converting trees back to graphs"):
        graph = Graph.load_file(input_path)
        tree = Tree()
        for node in graph.get_nodes():
            tree.add_node(node)
        for edge in graph.get_edges():
            tree.add_edge(edge)
        relative = input_path.relative_to(input_dir)
        output_graph = tree.revert_to_graph()
        output_path: Path = output_dir / relative
        output_graph.write_dot(output_path)
        with open(output_path, "w") as f:
            f.write(output_graph.to_json())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--input_dir", type=Path, help="Path to input graph directory"
    )
    arg_parser.add_argument(
        "-o", "--output_dir", type=Path, help="Path to output graph directory"
    )

    main(arg_parser.parse_args())
