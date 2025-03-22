from dataclasses import dataclass
from src.graphson.raw_node import NodeType


@dataclass(frozen=True)  # set frozen to create hash method
class EdgeType:
    src_type: NodeType
    dst_type: NodeType


OPTYPE_LOOKUP: dict[EdgeType, str] = {
    EdgeType(NodeType.PROCESS_LET, NodeType.PROCESS_LET): "Start_Processlet",
    EdgeType(NodeType.PROCESS_LET, NodeType.FILE): "Write",
    EdgeType(NodeType.PROCESS_LET, NodeType.IP_CHANNEL): "Write",
    EdgeType(NodeType.FILE, NodeType.PROCESS_LET): "Read",
    EdgeType(NodeType.IP_CHANNEL, NodeType.PROCESS_LET): "Read",
}
