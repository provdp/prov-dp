from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx

from src.algorithm.edge_metadata import OPTYPE_LOOKUP, EdgeType
from src.algorithm.utility import get_cycle
from src.algorithm.wrappers.graph import Graph

from .edge import Edge
from .node import Node
from ...graphson import RawEdge, RawNode, NodeType


@dataclass
class Marker:
    node_id: int
    height: int
    size: int
    path: str
    tree: "Tree"
    bucket: int | None


@dataclass
class NodeStats:
    degree: int
    depth: int
    height: int
    size: int


@dataclass
class TreeStats:
    # Aggregates
    heights: list[int]
    depths: list[int]
    sizes: list[int]
    degrees: list[int]

    # Totals
    height: int
    size: int
    degree: int
    diameter: int


class Tree(Graph):
    graph: Graph
    marked_nodes: list[Marker]
    stats: dict[
        str, list[float]
    ]  # Used to keep track of stats from within forked processes

    _nodes: dict[int, Node]
    _edges: dict[int, Edge]
    _incoming_lookup: dict[int, set[int]]  # node_id: set[edge_id]
    _outgoing_lookup: dict[int, set[int]]  # node_id: set[edge_id]
    _node_stats: dict[int, NodeStats]

    def __init__(
        self, input_graph: Graph | None = None
    ):
        if input_graph is None:
            graph = Graph()
        else:
            graph = input_graph
        # Sync data with graph
        self.graph = graph
        self.graph_id = graph.graph_id
        self.root_node_id = graph.root_node_id
        self._nodes = graph._nodes
        self._edges = graph._edges
        self._incoming_lookup = graph._incoming_lookup
        self._outgoing_lookup = graph._outgoing_lookup

        # Algorithm-specific fields
        self._node_stats = {}
        self.marked_nodes = []
        self.stats = {}

        if input_graph is not None:
            # Modify self to become a graph
            self.preprocess()

    def get_subtree(
        self, root_node_id: int, visited_node_ids: set[int] | None = None
    ) -> "Tree":
        """
        :param root_node_id: ID of the root node
        :param visited_node_ids: Accumulating list of node IDs that have already been visited
        :return: Subtree rooted at the given node
        """
        # Check if we've already computed this subtree
        visited_node_ids = visited_node_ids or set()

        # Create a new GraphWrapper object to store the accumulating tree
        subtree = Tree()
        root_node = self.get_node(root_node_id)
        subtree_root_node = deepcopy(root_node)
        subtree.add_node(subtree_root_node)

        # Mark the node as visited
        visited_node_ids.add(root_node_id)

        # BFS recursively
        for edge_id in self.get_outgoing_edge_ids(root_node_id):
            edge = self.get_edge(edge_id)
            next_node_id = edge.get_dst_id()
            if next_node_id in visited_node_ids:
                continue

            # Get the next subgraph, then add the connecting edge, and subgraph to the accumulating subgraph
            next_subgraph = self.get_subtree(edge.get_dst_id(), visited_node_ids)

            # Deep copy the graph components into the accumulating subgraph
            for new_node in next_subgraph.get_nodes():  # Nodes need to be added first
                subtree.add_node(deepcopy(new_node))

            # Add edge to the accumulating subgraph
            subtree.add_edge(deepcopy(edge))
            for new_edge in next_subgraph.get_edges():
                subtree.add_edge(deepcopy(new_edge))

        return subtree

    def init_node_stats(self, root_node_id: int, depth: int) -> None:
        # initialize tree stat lookup
        edges = self.get_outgoing_edge_ids(root_node_id)
        if len(edges) == 0:
            self._node_stats[root_node_id] = NodeStats(
                height=0, size=1, depth=depth, degree=0
            )
            return

        size = 1
        heights_of_subtrees = []
        for edge_id in edges:
            edge = self.get_edge(edge_id)
            dst_id = edge.get_dst_id()
            self.init_node_stats(dst_id, depth + 1)
            stats = self.get_node_stats(dst_id)
            size += stats.size
            heights_of_subtrees.append(stats.height)
        height = 1 + max(heights_of_subtrees)

        self._node_stats[root_node_id] = NodeStats(
            height=height, size=size, depth=depth, degree=len(edges)
        )

    def get_node_stats(self, node_id: int) -> NodeStats:
        return self._node_stats[node_id]

    # Step 1. Original graph
    def __original_graph(self) -> None:
        pass

    # Step 2. Remove self-referential edges and End_Processlet edges
    def __filter_edges(self) -> None:
        edges_to_remove = []
        for edge in self.get_edges():
            if (
                edge.get_src_id() == edge.get_dst_id()
                or edge.get_op_type() == "End_Processlet"
            ):
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            self.remove_edge(edge)

    # Step 3. Break cycles: Invert all outgoing edges from files/IPs
    def __invert_outgoing_file_edges(self) -> None:
        edges_to_invert = []
        for node in self.get_nodes():
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            edges_to_invert.extend(self.get_outgoing_edge_ids(node.get_id()))

        for edge_id in edges_to_invert:
            edge = self.get_edge(edge_id)
            assert edge.get_src_id() != edge.get_dst_id()

            self.remove_edge(
                edge
            )  # Remove then re-add edge to update adjacency lookups
            edge.invert()  # Flip source and destination
            self.add_edge(edge)

        # Assert there are no system resources w/ outgoing edges
        for node in self.get_nodes():
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            assert len(self.get_outgoing_edge_ids(node.get_id())) == 0

    # The graph is now a directed acyclic graph
    # Step 4. Remove lattice structure: Duplicate file/IP nodes for each incoming edge
    def __duplicate_file_ip_leaves(self) -> None:
        nodes = self.get_nodes().copy()
        for node in nodes:
            # Get incoming edges of original node
            incoming_edge_ids = self.get_incoming_edge_ids(node.get_id()).copy()

            # If this is a process, or if the file/ip has 0/1 incoming, just skip.
            if node.get_type() == NodeType.PROCESS_LET or len(incoming_edge_ids) < 2:
                continue

            # Duplicate node for each incoming edge
            for edge_id in incoming_edge_ids:
                # Create new node
                new_node_id = (
                    self.get_next_node_id()
                )  # Modify node_id -> to keep track of the original node_id
                new_node = deepcopy(node)
                new_node.set_id(new_node_id)
                self.add_node(new_node)

                # Point edge to the new node ID
                edge = self.get_edge(edge_id)
                self.remove_edge(edge)
                edge.set_dst_id(new_node_id)
                self.add_edge(edge)  # This function modifies the graphs' adjacency maps

            # Remove original node
            self.remove_node(node)

    # Step 5. Convert forest to a tree
    def __add_virtual_root(self) -> None:
        agent_id = self.get_nodes()[0].node.model_extra[
            "AGENT_ID"
        ]  # AgentID is always the same for a given graph in DARPA
        # Create root node
        raw_root_node = RawNode(
            _id=self.get_next_node_id(),
            TYPE=NodeType.VIRTUAL,
        )
        raw_root_node.model_extra["EXE_NAME"] = "VIRTUAL"
        raw_root_node.model_extra["CMD"] = "VIRTUAL"
        raw_root_node.model_extra["_label"] = "VIRTUAL"
        raw_root_node.model_extra["AGENT_ID"] = agent_id
        raw_root_node.model_extra["REF_ID"] = -1

        root_node = Node(raw_root_node)
        self.add_node(root_node)

        self.root_node_id = root_node.get_id()

        # Add disjoint trees to root's children
        for node in self.get_nodes():
            # If this is a virtual node, or if it's not a root node, skip

            non_self_cycle_incoming_edges = []
            for edge_id in self.get_incoming_edge_ids(node.get_id()):
                edge = self.get_edge(edge_id)
                if edge.get_src_id() == edge.get_dst_id():
                    continue
                non_self_cycle_incoming_edges.append(edge_id)

            if len(non_self_cycle_incoming_edges) > 0 or node is root_node:
                continue

            # Create edge from virtual root to subtree root
            self.add_edge(
                Edge(
                    RawEdge(
                        _id=self.get_next_edge_id(),
                        _outV=root_node.get_id(),
                        _inV=node.get_id(),
                        OPTYPE="FILE_EXEC",
                        _label="FILE_EXEC",
                        EVENT_START=0,
                        EVENT_START_STR=0,
                        REL_TIME_START=0,
                        REL_TIME_END=0,
                        EVENT_END=0,
                        EVENT_END_STR=0,
                        TIME_START=0,
                        TIME_END=0,
                        IS_ALERT=0,
                    )
                )
            )

    # Undo Tree to Graph process
    def __remove_virtual_components(self):
        edges_to_remove = []
        for edge in self.get_edges():
            src_node = self.get_node(edge.get_src_id())
            if src_node.get_type() == NodeType.VIRTUAL:
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            self.remove_edge(edge)
        node_ids_to_remove = []
        for node in self.get_nodes():
            if node.get_type() == NodeType.VIRTUAL:
                node_ids_to_remove.append(node)
        for node in node_ids_to_remove:
            self.remove_node(node)

    def __re_combine_resource_nodes(self):
        node_lookup: dict[str, list[Node]] = {}
        for node in self.get_nodes():
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            token = node.get_token()
            if token in node_lookup:
                node_lookup[token].append(node)
            else:
                node_lookup[token] = [node]

        for node_list in node_lookup.values():
            if len(node_list) == 1:
                continue
            # Since they're identical, arbitrarily select the first node
            chosen_one = node_list[0]
            for node in node_list[1:]:
                # Recall: resource nodes have no outgoing edges
                for incoming_edge_id in self.get_incoming_edge_ids(node.get_id()):
                    edge = self.get_edge(incoming_edge_id)
                    self.remove_edge(edge)
                    edge.set_dst_id(chosen_one.get_id())
                    self.add_edge(edge)
                self.remove_node(node)

    def __revert_resource_edge_direction(self):
        candidates_to_invert: list[int] = []
        for node in self.get_nodes():
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            node_id = node.get_id()
            # all resource nodes only have incoming edges
            assert len(self.get_outgoing_edge_ids(node_id)) == 0
            candidates_to_invert.extend(self.get_incoming_edge_ids(node_id))

        for edge_id in candidates_to_invert:
            edge = self.get_edge(edge_id)
            if "read" in edge.get_token().lower():
                self.remove_edge(edge)
                edge.invert()
                self.add_edge(edge)

    # Sanity check for the tree: Verify it is a valid tree
    def assert_valid_tree(self):
        # Find the root: a node with no incoming edges
        root_candidates = [
            node_id
            for node_id, edges in self._incoming_lookup.items()
            if len(edges) == 0
        ]

        # There must be exactly one root node
        assert len(root_candidates) == 1
        root = root_candidates[0]

        visited = set()

        def is_tree(node_id: int, path: list | None = None):
            if path is None:
                path = []
            node = self.get_node(node_id)
            path = path + [f"[{node_id}: {node.get_token()}]"]

            assert (
                node_id not in visited
            ), f"Found a cycle: {get_cycle(path)}. Incoming: {[self.get_edge(e).get_token() for e in self.get_incoming_edge_ids(node_id)]}"

            visited.add(node_id)
            for edge_id in self.get_outgoing_edge_ids(node_id):
                edge = self.get_edge(edge_id)
                next_node_id = edge.get_dst_id()

                if not is_tree(next_node_id, path + [f"--{edge.get_token()}->"]):
                    return False
            return True

        # Start DFS from root to check for cycles and if all nodes are reachable
        assert is_tree(root), "Not all nodes are reachable"

        # Check if all nodes were visited (tree is connected)
        assert len(visited) == len(
            self._nodes
        ), f"Visited {len(visited)}/{len(self._nodes)}, tree is NOT connected"

    __conversion_steps: list[tuple[Callable | None, Callable | None]] = [
        (__original_graph, None),
        (__filter_edges, None),  # Edge filter cannot be undone
        (__invert_outgoing_file_edges, __revert_resource_edge_direction),
        (__duplicate_file_ip_leaves, __re_combine_resource_nodes),
        (__add_virtual_root, __remove_virtual_components),
        (None, __original_graph),
    ]

    def __process_pipeline(
        self, steps: list[Callable | None], output_dir: Path | None = None
    ):
        for i, step in enumerate(steps):
            if step is None:
                continue
            step(self)
            step_name = f'{i + 1}_{step.__name__.strip("_")}'
            if output_dir is not None:
                with open(
                    output_dir / f"{step_name}.json", "w", encoding="utf-8"
                ) as output_file:
                    output_file.write(self.to_json())
                self.to_dot().save(output_dir / f"{step_name}.dot")

    def preprocess(self, output_dir: Path | None = None) -> "Tree":
        self.__process_pipeline(
            [step for step, _ in self.__conversion_steps], output_dir
        )
        return self

    def revert_to_graph(self, output_dir: Path | None = None) -> Graph:
        self.__process_pipeline(
            [step for _, step in reversed(self.__conversion_steps)], output_dir
        )
        return self.graph

    def prune_tree(self, root_node_id: int) -> "Tree":
        # Create subtree graph
        subtree: Tree = self.get_subtree(root_node_id)
        subtree.graph_id = self.graph_id
        num_roots = 0
        # Remove all subtree nodes and elements from the parent graph
        for edge in subtree.get_edges():
            self.remove_edge(edge)
        for node in subtree.get_nodes():
            if node.get_id() == root_node_id:
                num_roots += 1
                continue  # We want to keep this node, so we can replace later
            self.remove_node(node)

        # Add the root edge and parent to the subtree, so we can preserve the edge-node relationship
        # Make sure this happens after removing nodes, so we don't remove the edge and its parent
        incoming_edge_ids = self.get_incoming_edge_ids(root_node_id)
        assert (
            len(incoming_edge_ids) == 1
        ), f"Pruned tree should have 1 incoming edge, found {len(incoming_edge_ids)}"
        root_edge_id = incoming_edge_ids[0]
        root_edge = self.get_edge(root_edge_id)
        root_edge_source = self.get_node(root_edge.get_src_id())
        subtree.add_node(root_edge_source)
        subtree.add_edge(root_edge)

        # Sanity checks on the tree's state
        assert num_roots == 1, f"Expected 1 root, got {num_roots}"

        assert (
            len(self.get_incoming_edge_ids(root_node_id)) == 1
        ), f"Expected 1 outgoing edge, got {len(self.get_outgoing_edge_ids(root_node_id))}"
        subtree_root = subtree.get_node(root_node_id)
        assert subtree_root is not None
        assert len(subtree.get_incoming_edge_ids(root_node_id)) == 1

        return subtree

    def path_to_string(self, path: list[int]) -> str:
        tokens = []
        for edge_id in path:
            edge = self.get_edge(edge_id)
            node = self.get_node(edge.get_src_id())
            tokens.extend([node.get_token(), edge.get_token()])

        return " ".join(tokens)

    def replace_node_with_tree(self, node_id_to_replace: int, graph: "Tree") -> None:
        """
        Attach a subtree to the destination of the given edge.
        ex: X is the node to replace.
            A -e-> X
        The tree's root MUST have out-degree 0, so it can be represented as this:
            R -f-> T
        e is replaced with f, X with T, and A is ignored. Result:
            A -f-> T

        NOTE: If R is a virtual node, then we update edge -f->
        features arbitrarily to make it a legal edge

        @param node_id_to_replace: node to replace with subtree
        @param graph: subtree to replace with
        """
        # self.assert_valid_tree()
        node_id_translation = {}
        edge_id_translation = {}
        # Update node IDs to avoid collision in the current graph
        orphan_nodes = []
        for old_node in graph.get_nodes():
            # Copy the node, and give it a new ID
            old_node_id = old_node.get_id()
            node = deepcopy(old_node)
            node_id = self.get_next_node_id()
            node.set_id(node_id)

            # If the node is an orphan, it's the root R, so keep track of it.
            if len(graph.get_incoming_edge_ids(old_node_id)) == 0:
                orphan_nodes.append(node)

            # Add the ID to the lookup
            assert node_id_translation.get(old_node_id) is None
            node_id_translation[old_node_id] = node_id
            self.add_node(node)

            # Mark the node to indicate it's been added after the fact
            node.marked = True

        # There should only be one orphan/root node (R)
        assert (
            len(orphan_nodes) == 1
        ), f"Expected 1 orphan node, got {len(orphan_nodes)}/{len(graph.get_nodes())}"
        R: Node = orphan_nodes[0]

        # Update edge IDs in the subtree to avoid collision in the current graph, and bring up to date with node IDs
        for old_edge in graph.get_edges():
            # Copy the edge, and give it a new ID
            edge = deepcopy(old_edge)
            new_edge_id = self.get_next_edge_id()
            edge.set_id(new_edge_id)
            # Update the edge's node IDs to match the new graph
            assert node_id_translation.get(edge.get_src_id()) is not None
            assert node_id_translation.get(edge.get_dst_id()) is not None
            edge.translate_node_ids(node_id_translation)
            assert self.get_node(edge.get_src_id()) is not None
            assert self.get_node(edge.get_dst_id()) is not None

            # Add the ID to the lookup
            edge_id_translation[old_edge.get_id()] = new_edge_id
            self.add_edge(edge)
            assert new_edge_id in self.get_outgoing_edge_ids(edge.get_src_id())
            assert new_edge_id in self.get_incoming_edge_ids(edge.get_dst_id())
            # Mark the edge to indicate it's added after the fact
            edge.marked = True

        # A -e-> X  original self
        # R -f-> T  tree to add
        # A -f-> T  new self

        # find f
        outgoing_edges = [
            self.get_edge(e) for e in self.get_outgoing_edge_ids(R.get_id())
        ]
        assert len(outgoing_edges) == 1
        edge_f = outgoing_edges[0]

        # Remove -e-> X from the graph
        X = self.get_node(node_id_to_replace)
        incoming_edges = self.get_incoming_edge_ids(node_id_to_replace)
        assert len(incoming_edges) == 1
        edge_e_id = incoming_edges[0]
        edge_e = self.get_edge(edge_e_id)
        self.remove_edge(edge_e)  # Remove edge before node to preserve graph state
        self.remove_node(X)

        # Remove f so we can modify it. Must happen before removing R to preserve graph state
        self.remove_edge(edge_f)
        # Remove R from the graph
        self.remove_node(R)

        # -f-> T is already in the graph, so attach it to A and we're done
        A_id = edge_e.get_src_id()
        edge_f.set_src_id(A_id)
        self.add_edge(edge_f)  # Add f back into the graph
        assert edge_f.get_id() in self.get_outgoing_edge_ids(edge_e.get_src_id())

        # If R was a virtual node and A isn't, -f-> is an invalid edge. Update features to be legal
        A: Node = self.get_node(A_id)
        if R.get_type() == NodeType.VIRTUAL and A.get_type() != NodeType.VIRTUAL:
            self.__update_edge_attributes(edge_f)

        self.assert_valid_tree()

    def __update_edge_attributes(self, edge: Edge) -> None:
        src = self.get_node(edge.get_src_id())
        src_type = src.get_type()
        assert (
            src_type == NodeType.PROCESS_LET
        ), f"Trees should only be reattached to processlet nodes, found {src_type}"
        dst = self.get_node(edge.get_dst_id())
        dst_type = dst.get_type()
        edge_type = EdgeType(src_type=NodeType.PROCESS_LET, dst_type=dst_type)
        assert dst_type in [
            NodeType.PROCESS_LET,
            NodeType.FILE,
            NodeType.IP_CHANNEL,
        ], f"Unexpected dst node type: {dst_type}"
        edge.edge.optype = OPTYPE_LOOKUP[edge_type]

    def size(self) -> int:
        return len(self._nodes)

    def add_stat(self, stat: str, value: float):
        if stat not in self.stats:
            self.stats[stat] = []
        self.stats[stat].append(value)

    def get_stats(self) -> "TreeStats":
        self._node_stats = {}  # HACK:  this is not good
        self.init_node_stats(self.get_root_id(), 0)
        self.assert_valid_tree()
        heights = []
        depths = []
        sizes = []
        degrees = []
        G = self.to_nx().to_undirected()
        diameter = max([max(j.values()) for (_, j) in nx.shortest_path_length(G)])
        del G

        assert len(self._node_stats) == len(
            self._nodes
        ), f"{len(self._node_stats)}, {len(self._nodes)}"
        for node_id in self._nodes.keys():
            assert node_id in self._nodes, f"Node {node_id} doesnt exist"
            assert node_id in self._node_stats, self.get_incoming_edge_ids(node_id)
            stat = self.get_node_stats(node_id)
            heights.append(stat.height)
            depths.append(stat.depth)
            sizes.append(stat.size)
            degrees.append(len(self.get_outgoing_edge_ids(node_id)))

        return TreeStats(
            heights=heights,
            depths=depths,
            sizes=sizes,
            degrees=degrees,
            height=max(heights),
            size=max(sizes),
            degree=max(degrees),
            diameter=diameter,
        )
