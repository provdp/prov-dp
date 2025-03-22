from collections import deque
import json
from pathlib import Path

import graphviz as gv
import networkx as nx

from .edge import Edge
from .node import Node
from ...graphson import RawEdge, RawNode, RawGraph


class Graph:
    graph_id: int | None
    root_node_id: int | None

    _nodes: dict[int, Node]
    _edges: dict[int, Edge]
    _incoming_lookup: dict[int, set[int]]  # node_id: set[edge_id]
    _outgoing_lookup: dict[int, set[int]]  # node_id: set[edge_id]

    @classmethod
    def load_file(cls, path: Path) -> "Graph":
        file_name = str(path.stem)
        if "-" in file_name:
            split = file_name.split("-")
        elif "_" in file_name:
            split = file_name.split("_")
        else:
            raise ValueError(f"Invalid file name: {file_name}")
        ref_id = -1
        if len(split) == 3:
            ref_id = int(split[1])

        return Graph(RawGraph.load_file(path), source_edge_ref_id=ref_id)

    def __init__(
        self, graph: RawGraph | None = None, source_edge_ref_id: int | None = None
    ):
        graph = graph or RawGraph()
        self._incoming_lookup = {}
        self._outgoing_lookup = {}
        self.__init_nodes(graph.nodes)
        self.__init_edges(graph.edges)
        self.__init_source(source_edge_ref_id)

    def __init_nodes(self, nodes: list[RawNode]):
        # Create a lookup by node ID
        self._nodes = {}
        for raw_node in nodes:
            self.add_node(Node(raw_node))

    def __init_edges(self, edges: list[RawEdge]):
        # Create a lookup by edge ID and add edge references to nodes
        self._edges = {}
        for raw_edge in edges:
            self.add_edge(Edge(raw_edge))

    def __init_source(self, source_edge_ref_id: int | None) -> None:
        # Set the graph_id to keep track of the original graph
        self.graph_id = source_edge_ref_id
        if source_edge_ref_id is not None:
            # Ref ID is not the same as graphson ID, so we need to find the edge with the matching ref ID
            matches = [
                edge
                for edge in self._edges.values()
                if edge.get_ref_id() == source_edge_ref_id
            ]
            if len(matches) == 1:
                self.root_node_id = matches[0].get_src_id()
                return
        self.root_node_id = None

    # Wrapper functions
    def get_edges(self) -> list[Edge]:
        return list(self._edges.values())

    def get_nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def add_edge(self, edge: Edge) -> None:
        assert self._edges.get(edge.get_id()) is None
        assert (
            self.get_node(edge.get_src_id()) is not None
        ), f"Edge {edge.get_id()} has no source in graph"
        assert (
            self.get_node(edge.get_dst_id()) is not None
        ), f"Edge {edge.get_id()} has no destination in graph"
        edge_id = edge.get_id()

        # Add edge to graph and lookup
        self._edges[edge_id] = edge
        self._incoming_lookup[edge.get_dst_id()].add(edge_id)
        self._outgoing_lookup[edge.get_src_id()].add(edge_id)

    def add_node(self, node: Node) -> None:
        node_id = node.get_id()
        assert self._nodes.get(node_id) is None
        self._nodes[node_id] = node
        self._incoming_lookup[node_id] = set()
        self._outgoing_lookup[node_id] = set()

    def remove_node(self, node: Node) -> None:
        # Removes node from graph and lookup
        node_id = node.get_id()
        assert self._nodes.get(node_id) is not None
        self._nodes.pop(node_id)
        self._incoming_lookup.pop(node_id)
        self._outgoing_lookup.pop(node_id)

    def remove_edge(self, edge: Edge) -> None:
        # Removes edge from graph and lookup
        edge_id = edge.get_id()
        self._edges.pop(edge_id)
        self._incoming_lookup[edge.get_dst_id()].remove(edge_id)
        self._outgoing_lookup[edge.get_src_id()].remove(edge_id)

    def get_next_node_id(self) -> int:
        return max([node_id for node_id in self._nodes.keys()]) + 1

    def get_next_edge_id(self) -> int:
        return max([edge_id for edge_id in self._edges.keys()]) + 1

    def get_outgoing_edge_ids(self, node_id: int) -> list[int]:
        return list(self._outgoing_lookup[node_id])

    def get_incoming_edge_ids(self, node_id: int) -> list[int]:
        return list(self._incoming_lookup[node_id])

    def get_node(self, node_id: int) -> Node:
        node = self._nodes.get(node_id)
        assert node is not None, f"Node {node_id} does not exist"
        return node

    def get_edge(self, edge_id: int) -> Edge:
        edge = self._edges.get(edge_id)
        assert edge is not None, f"Edge {edge_id} does not exist"
        return edge

    def has_edge(self, src_id: int, dst_id: int) -> bool:
        return dst_id in self.get_outgoing_edge_ids(src_id)

    def clear(self) -> None:
        root_id = self.get_root_id()  # Need to call before removing edges

        for edge in self.get_edges():
            self.remove_edge(edge)
        assert len(self._edges) == 0

        for node in self.get_nodes():
            if node.get_id() == root_id:
                continue
            self.remove_node(node)
        assert len(self._nodes) <= 1

    def remove_disconnected_components(self) -> None:
        """
        Traverse graph starting from root.
        Remove all nodes/edges that cannot be reached
        """
        untouched_edge_ids: set[int] = set(self._edges.keys())
        untouched_node_ids: set[int] = set(self._nodes.keys())

        assert self.root_node_id is not None
        queue: deque[int] = deque([self.root_node_id])
        while len(queue) > 0:
            node_id = queue.popleft()
            if node_id not in untouched_node_ids:
                continue
            untouched_node_ids.remove(node_id)

            # Add parent nodes
            for edge_id_in in self.get_incoming_edge_ids(node_id):
                if edge_id_in not in untouched_edge_ids:
                    continue
                untouched_edge_ids.remove(edge_id_in)
                queue.append(self.get_edge(edge_id_in).get_src_id())

            # Add child nodes
            for edge_id_out in self.get_outgoing_edge_ids(node_id):
                if edge_id_out not in untouched_edge_ids:
                    continue
                untouched_edge_ids.remove(edge_id_out)
                queue.append(self.get_edge(edge_id_out).get_dst_id())

        # Remove all edges that haven't been touched
        for edge_id in untouched_edge_ids:
            self.remove_edge(self.get_edge(edge_id))
        # Now remove all untouched nodes
        for node_id in untouched_node_ids:
            self.remove_node(self.get_node(node_id))

    # Exporter functions
    def to_dot(self) -> gv.Digraph:
        dot_graph = gv.Digraph()
        dot_graph.attr(rankdir="LR")
        included_nodes: set[Node] = set()
        sorted_edges = sorted(self.get_edges(), key=lambda e: e.get_time())

        def add_to_graph(new_node: Node):
            assert new_node is not None, "Trying to add a null node to the graph"
            included_nodes.add(new_node)
            dot_graph.node(str(new_node.get_id()), **new_node.to_dot_args())

        num_missing = 0
        num_null = 0
        for edge in sorted_edges:
            src_id, dst_id = edge.get_src_id(), edge.get_dst_id()
            assert src_id is not None, f"Edge {edge.get_id()} has no source"
            assert dst_id is not None, f"Edge {edge.get_id()} has no destination"
            src, dst = self.get_node(src_id), self.get_node(dst_id)
            add_to_graph(src)
            add_to_graph(dst)

            dot_graph.edge(str(src_id), str(dst_id), **edge.to_dot_args())

        if num_missing > 0:
            print(
                f"Warn: {num_missing} MIA, {num_null} null out of {len(self.get_edges())}?"
            )
        for node in self.get_nodes():
            if node not in included_nodes:
                add_to_graph(node)

        return dot_graph

    def write_dot(self, file_path: Path, to_pdf = False) -> None:
        file_path = file_path.with_suffix(".dot")
        file_path.parent.mkdir(exist_ok=True, parents=True)
        dot_graph = self.to_dot()
        dot_graph.save(file_path)
        if to_pdf:
            dot_graph.render(file_path, format="pdf")

    def to_nx(self) -> nx.DiGraph:
        digraph: nx.DiGraph = nx.DiGraph()
        # NetworkX node IDs must index at 0
        node_ids = {node.get_id(): i for i, node in enumerate(self.get_nodes())}
        for node in self.get_nodes():
            digraph.add_node(node_ids[node.get_id()], feature=node.get_token())
        for edge in self.get_edges():
            src, dst = edge.get_src_id(), edge.get_dst_id()
            if src is not None and dst is None:
                continue
            digraph.add_edge(node_ids[src], node_ids[dst], feature=edge.get_token())
        return digraph

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": "EXTENDED",
                "vertices": [node.to_json_dict() for node in self.get_nodes()],
                "edges": [edge.to_json_dict() for edge in self.get_edges()],
            }
        )

    def get_root_id(self) -> int:
        root_ids = [
            node_id
            for node_id in self._nodes.keys()
            if len(self.get_incoming_edge_ids(node_id)) == 0
        ]
        assert len(root_ids) == 1, f"Expected only 1 root, got {len(root_ids)}"
        return root_ids[0]

    def assert_complete(self) -> None:
        for edge in self.get_edges():
            assert (
                edge.get_src_id() is not None
            ), f"Edge {edge.get_id()} ({edge.get_token()} has None source"
            assert (
                edge.get_dst_id() is not None
            ), f"Edge {edge.get_id()} ({edge.get_token()} has None destination"
            if self.get_node(edge.get_src_id()) is None:
                print(f"Edge {edge.get_id()} ({edge.get_token()}) has no source")
            assert (
                self.get_node(edge.get_dst_id()) is not None
            ), f"Edge {edge.get_id()} ({edge.get_token()}) has no destination"
        for node in self.get_nodes():
            node_id = node.get_id()
            assert node.get_id() is not None, f"Node {node.get_token()} has None ID"
            for edge_id in self.get_incoming_edge_ids(node_id):
                edge = self.get_edge(edge_id)
                assert edge.get_dst_id() == node_id, (
                    f"Node {node_id} has incoming edge {edge_id} "
                    f"with wrong destination ({edge.get_src_id()} -> {edge.get_dst_id()})"
                )
            for edge_id in self.get_outgoing_edge_ids(node_id):
                edge = self.get_edge(edge_id)
                assert (
                    edge.get_src_id() == node_id
                ), f"Node {node.get_token()} has outgoing edge {edge_id} with wrong source"
