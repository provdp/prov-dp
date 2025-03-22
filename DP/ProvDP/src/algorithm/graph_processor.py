from dataclasses import dataclass
import gc
import pickle
import random
from collections import deque
from pathlib import Path
from typing import Any, Callable, Generator, Iterable

import numpy as np
from tqdm import tqdm

from src.algorithm.wrappers.graph import Graph
from src.algorithm.wrappers.tree import Marker, TreeStats

from .utility import print_stats, logistic_function, smart_map, RANDOM_SEED
from .wrappers import Tree

PRUNED_TREE_SIZE = "pruned tree size (#nodes)"
PRUNED_TREE_HEIGHT = "pruned tree height"
PRUNED_TREE_DEPTH = "pruned tree depth"
NUM_MARKED_NODES = "# marked nodes"
ATTACHED_TREE_SIZE = "attached tree size (#nodes)"
NUM_UNMOVED_SUBTREES = "# unmoved subtrees"
NUM_UNCHANGED_SUBTREES = "# unchanged subtrees"
PERCENT_UNMOVED_SUBTREES = "% unmoved subtrees"
REATTACH_PROBAILITIES = "reattach probabilities"


@dataclass
class ReattachData:
    epsilon_2: float
    tree: Tree
    size_array: np.ndarray
    count_array: np.ndarray


@dataclass
class GraphStats:
    node_stats: dict
    tree_stats: dict


class GraphProcessor:
    # Pruning parameters
    __epsilon_1: float
    __epsilon_2: float
    __alpha: float
    __beta: float
    __gamma: float
    __eta: float
    __k: int

    # List to aggregate training data (path: str, subtree: Tree) tuples
    __pruned_subtrees: list[Marker]

    # Processing pipeline
    __single_threaded: bool

    # Step labels
    __step_number: int

    # Checkpoint flags
    __load_perturbed_graphs: bool

    # Stats (each stat contains a list of samples)
    stats: dict[str, list[float]]

    def __init__(
        self,
        epsilon: float,
        delta: float,
        alpha: float,
        beta: float,
        gamma: float,
        eta: float,
        k: int,
        output_dir: Path = Path("."),
        single_threaded: bool = False,
        load_perturbed_graphs: bool = False,
    ):
        total = alpha + beta + gamma + eta
        assert abs(total - 1) < 0.000001, f"Hyperparameters must sum to 1, got {total}"
        # Seed
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        # Pruning parameters
        epsilon_1 = epsilon * delta
        epsilon_2 = epsilon * (1 - delta)
        print(f"ARGUMENTS: epsilon={epsilon}, delta={delta}, alpha={alpha}, beta={beta}, gamma={gamma} eta={eta}")
        print(f"ARGUMENTS: epsilon={epsilon}, delta={delta} (e_1={epsilon_1}, e_2={epsilon_2})")
        self.__epsilon_1 = epsilon_1
        self.__epsilon_2 = epsilon_2
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__eta = eta
        self.__k = k

        # List to aggregate training data
        self.__pruned_subtrees = []

        # Logging
        self.__step_number = 0

        # argparse args
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Algorithm configuration
        self.__single_threaded = single_threaded

        # Checkpoint flags
        self.__load_perturbed_graphs = load_perturbed_graphs

        # Stats
        self.stats = {}

    def __step(self) -> str:  # Step counter for pretty logging
        self.__step_number += 1
        return f"({self.__step_number})"

    def __map(
        self, func: Callable[[Any], Any], items: Iterable[Any], desc: str = ""
    ) -> Generator:
        generator = smart_map(
            func=func, items=items, single_threaded=self.__single_threaded, desc=desc
        )
        for item in generator:
            yield item

    @classmethod
    def load_tree_from_file(cls, path: Path) -> Tree:
        graph = Graph.load_file(path)
        tree = Tree(graph)
        tree.assert_valid_tree()

        return tree

    def preprocess_graphs(self, paths: list[Path]) -> list[Tree]:
        trees = list(
            self.__map(self.load_tree_from_file, paths, "Preprocessing graphs")
        )
        # print("Original Tree stats:")
        # self.print_tree_stats(trees)
        return trees

    def get_graph_stats(self, trees: list[Tree]) -> GraphStats:
        stats: list[TreeStats] = list(
            smart_map(
                func=Tree.get_stats,
                items=trees,
                single_threaded=self.__single_threaded,
                desc="Calculating stats",
            )
        )

        node_stats = {"heights": [], "depths": [], "sizes": [], "degrees": []}

        tree_stats = {"heights": [], "sizes": [], "degrees": [], "diameters": []}
        for stat in tqdm(stats, desc="Aggregating stats"):
            # Node stats
            node_stats["heights"].extend(stat.heights)
            node_stats["depths"].extend(stat.depths)
            node_stats["sizes"].extend(stat.sizes)
            node_stats["degrees"].extend(stat.degrees)

            # Tree stats
            tree_stats["heights"].append(stat.height)
            tree_stats["sizes"].append(stat.size)
            tree_stats["degrees"].append(stat.degree)
            tree_stats["diameters"].append(stat.diameter)

        return GraphStats(node_stats=node_stats, tree_stats=tree_stats)

    def print_tree_stats(self, trees: list[Tree]):
        graph_stats = self.get_graph_stats(trees)
        tree_stats = graph_stats.tree_stats
        print_stats("Tree height", tree_stats["heights"])
        print_stats("Tree size", tree_stats["sizes"])
        print_stats("Degrees", tree_stats["degrees"])
        print_stats("Diameters", tree_stats["diameters"])

    def load_and_prune_graphs(self, paths: list[Path]) -> list[Tree]:
        # Try to load checkpoint if one exists
        pruned_graph_path = self.output_dir / "pruned_graphs.pkl"
        if self.__load_perturbed_graphs and pruned_graph_path.exists():
            # Load graphs and training data from file
            print(
                f"{self.__step()} Loading pruned graphs and training data from {pruned_graph_path}"
            )
            with open(pruned_graph_path, "rb") as f:
                pruned_graphs, pruned_subtrees = pickle.load(f)
                self.__pruned_subtrees = pruned_subtrees
                print(
                    f"  Loaded {len(pruned_graphs)} graphs and {len(pruned_subtrees)} training samples"
                )
                return pruned_graphs

        # Load and convert input graphs to trees
        trees: list[Tree] = self.preprocess_graphs(paths)

        pruned_trees: list[Tree] = list(
            self.__map(self.prune, trees, f"{self.__step()} Pruning graphs")
        )

        # Aggregate training data from trees
        self.__pruned_subtrees: list[Marker] = []
        for tree in pruned_trees:
            self.__pruned_subtrees.extend(tree.marked_nodes)
            self.__add_stats(tree.stats)

        # self.__print_stats()

        # Write result to checkpoint
        with open(pruned_graph_path, "wb") as f:
            # Save a (pruned_graphs, training_data) tuple
            pickle.dump((pruned_trees, self.__pruned_subtrees), f)
            print(
                f"  Wrote {len(pruned_trees)} graphs to {pruned_graph_path}"
            )

        return pruned_trees

    def prune(self, tree: Tree) -> Tree:
        # Breadth first search through the graph, keeping track of the path to the current node
        # (node_id, list[edge_id_path]) tuples
        root_node_id = tree.get_root_id()
        tree.init_node_stats(root_node_id, 0)
        queue: deque[tuple[int, list[int]]] = deque([(root_node_id, [])])
        visited_node_ids: set[int] = set()
        subtrees_pruned = 0
        while len(queue) > 0 and subtrees_pruned < self.__k:
            # Standard BFS operations
            src_node_id, path = queue.popleft()

            if src_node_id in visited_node_ids:
                continue
            visited_node_ids.add(src_node_id)

            # calculate the probability of pruning a given tree
            node_stats = tree.get_node_stats(src_node_id)
            subtree_size, height, depth, degree = (
                node_stats.size,
                node_stats.height,
                node_stats.depth,
                node_stats.degree,
            )
            # assert depth == len(path)
            distance = (
                (self.__alpha * subtree_size)
                + (self.__beta * height)
                + (self.__gamma * depth)
                + (self.__eta * degree)
            )
            p = logistic_function(
                self.__epsilon_1 * distance
            )  # big distance -> lower probability of pruning
            prune_edge: bool = np.random.choice([True, False], p=[p, 1 - p])
            # if we prune, don't add children to queue
            if prune_edge:
                if len(path) == 0:
                    # If this is the root, remove the entire tree from the dataset.
                    tree.clear()
                    return tree

                # remove the tree rooted at this edge's dst_id from the graph
                pruned_tree = tree.prune_tree(src_node_id)
                # Keep track of the node and its path, so we can attach to it later
                path_string = tree.path_to_string(path)

                # Mark the node and keep its stats
                tree.marked_nodes.append(
                    Marker(
                        node_id=src_node_id,
                        height=height,
                        size=subtree_size,
                        path=path_string,
                        tree=pruned_tree,
                        bucket=None,
                    )
                )

                # ensure we don't try to bfs into the pruned tree
                visited_node_ids.update(
                    node.get_id() for node in pruned_tree.get_nodes()
                )

                # track statistics
                tree.add_stat(PRUNED_TREE_SIZE, subtree_size)
                tree.add_stat(PRUNED_TREE_HEIGHT, height)
                tree.add_stat(PRUNED_TREE_DEPTH, depth)

                continue

            # otherwise, continue adding children to the BFS queue
            for edge_id in tree.get_outgoing_edge_ids(src_node_id):
                edge = tree.get_edge(edge_id)
                dst_node_id = edge.get_dst_id()
                queue.append((dst_node_id, path + [edge_id]))

        return tree

    def perturb_graphs(self, paths: list[Path]) -> list[Tree]:
        pruned_graphs: list[Tree] = self.load_and_prune_graphs(paths)
        # Read graphs and run pruning step

        gc.collect()  # Pray memory usage goes down
        # Run grafting
        self.__re_add_with_bucket(pruned_graphs)

        for tree in pruned_graphs:
            tree.assert_valid_tree()

        if len(self.stats.get(NUM_UNMOVED_SUBTREES, [])) > 0:
            num_unmoved_subtrees = self.stats[NUM_UNMOVED_SUBTREES]
            num_marked_nodes = self.stats[NUM_MARKED_NODES]
            self.stats[PERCENT_UNMOVED_SUBTREES] = [
                (x / max(y, 0.0001)) * 100
                for x, y in zip(num_unmoved_subtrees, num_marked_nodes)
            ]
            
        # self.__print_stats()
        # print("Perturb Tree stats:")
        # self.print_tree_stats(pruned_graphs)

        # Revert trees to graphs
        for tree in pruned_graphs:
            tree.revert_to_graph()

        return pruned_graphs

    def __re_add_with_bucket(self, pruned_trees: list[Tree]):
        buckets: dict[int, list[Tree]] = {}
        for marker in self.__pruned_subtrees:
            size = marker.size
            if size not in buckets:
                buckets[size] = []
            buckets[size].append(marker.tree)

        print("Bucket sizes:")
        for size in sorted(buckets.keys()):
            print(f"  size {size}: {len(buckets[size])} subtrees")

        size_array = np.array([size for size in buckets.keys()])
        count_array = np.array([len(bucket) for _, bucket in buckets.items()])
        assert len(size_array) == len(buckets)

        for tree in tqdm(pruned_trees, desc=f"{self.__step()} Re-attaching subgraphs"):
            # Stats
            self.__add_stat(NUM_MARKED_NODES, len(tree.marked_nodes))
            unmoved_subtrees = 0
            unchanged_subtrees = 0

            for marker in tree.marked_nodes:
                perturbed_size = round(
                    marker.size + np.random.laplace(0, 1 / self.__epsilon_2)
                )
                spread = 1  # low spread -> more uniform distance distribution -> more uniform probability -> less likely to choose tree w/ matching size
                distances = (abs(size_array - perturbed_size) + 1) ** spread

                unscaled_weights = 1 / distances
                weights = np.multiply(
                    unscaled_weights, count_array
                )  # Scale weight by corresponding bucket size (pair-wise multiplication)

                probabilities = weights / sum(weights)
                # (1) Choose bucket with probability proportional to bucket size, inversely proportional to difference in size
                size_choice = np.random.choice(size_array, p=probabilities)
                bucket_choice: list[Tree] = buckets[size_choice]

                subtree: Tree = random.choice(
                    bucket_choice
                )  # (2) Choose uniformly from bucket

                tree.replace_node_with_tree(marker.node_id, subtree)

                # Stats
                # Recall: pruned subtrees have an additional node
                self.__add_stat(ATTACHED_TREE_SIZE, (subtree.size() - 1))

                # If the graph ID is the same, this subtree is in the same graph.
                if subtree.graph_id == tree.graph_id:
                    unmoved_subtrees += 1

                    # If the root ID is the same, this subtree was effectively not moved.
                    subtree_root = subtree.get_node(subtree.get_root_id())
                    if subtree_root.get_id() == marker.node_id:
                        unchanged_subtrees += 1

            # Stats
            self.__add_stat(NUM_UNMOVED_SUBTREES, unmoved_subtrees)
            self.__add_stat(NUM_UNCHANGED_SUBTREES, unmoved_subtrees)

    def __print_stats(self):
        for stat, values in self.stats.items():
            print_stats(stat, values)

    def __add_stat(self, stat: str, value: float):
        if stat not in self.stats:
            self.stats[stat] = []
        self.stats[stat].append(value)

    def __add_stats(self, stats: dict[str, list[float]]):
        for stat, values in stats.items():
            if stat not in self.stats:
                self.stats[stat] = []
            self.stats[stat].extend(values)
