import json
from pathlib import Path
from pydantic import BaseModel, Field

from .raw_edge import RawEdge
from .raw_node import RawNode


# noinspection PyArgumentList
class RawGraph(BaseModel):
    nodes: list[RawNode] = Field(alias="vertices", default_factory=list)
    edges: list[RawEdge] = Field(alias="edges", default_factory=list)

    @staticmethod
    def load_file(path_to_json: Path):
        with open(path_to_json, "r", encoding="utf-8") as input_file:
            input_json = json.load(input_file)
            return RawGraph(**input_json)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict:
        model = self.model_dump(by_alias=True)
        model["mode"] = "EXTENDED"
        model["vertices"] = [node.to_dict() for node in self.nodes]
        model["edges"] = [edge.to_dict() for edge in self.edges]
        return model
