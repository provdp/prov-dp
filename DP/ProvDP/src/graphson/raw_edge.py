from pydantic import Field

from .graphsonobject import GraphsonObject

EDGE_ID_SEQUENCE = 5000


class RawEdge(GraphsonObject):
    id: int = Field(alias="_id")
    src_id: int = Field(..., alias="_outV")
    dst_id: int = Field(..., alias="_inV")
    optype: str = Field(..., alias="OPTYPE")
    label: str = Field(..., alias="_label")
    time: int = Field(..., alias="EVENT_START")

    def __repr__(self):
        return f"{self.src_id}-{self.optype}-{self.dst_id}"

    @staticmethod
    def of(
        src_id: int,
        dst_id: int,
        optype: str,
        time: int,
        new_id: int = None,
    ):
        global EDGE_ID_SEQUENCE
        if new_id is None:
            new_id = EDGE_ID_SEQUENCE
            EDGE_ID_SEQUENCE += 1
        return RawEdge(
            _id=new_id, _inV=src_id, _outV=dst_id, OPTYPE=optype, EVENT_START=time
        )

    def __hash__(self):
        return hash((self.src_id, self.dst_id))
