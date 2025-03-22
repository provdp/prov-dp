from enum import Enum

from pydantic import Field

from .graphsonobject import GraphsonObject


class NodeType(Enum):
    PROCESS_LET = "ProcessNode"
    FILE = "FileNode"
    IP_CHANNEL = "SocketChannelNode"
    VIRTUAL = "VirtualNode"

    def __str__(self):
        return self.name

    def __int__(self):
        raise ValueError


class RawNode(GraphsonObject):
    id: int = Field(..., alias="_id")
    type: NodeType = Field(..., alias="TYPE")
    marked: bool = False

    def __hash__(self) -> int:
        return hash(self.id)
