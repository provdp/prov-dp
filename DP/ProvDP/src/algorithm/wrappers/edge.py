from ..utility import json_value
from ...graphson import RawEdge


class Edge:
    edge: RawEdge
    marked: bool = False

    def __init__(self, edge: RawEdge):
        self.edge = edge

    def invert(self) -> None:
        # Swap the src and dst ids
        src_id, dst_id = self.get_src_id(), self.get_dst_id()
        self.set_src_id(dst_id)
        self.set_dst_id(src_id)

    def get_id(self) -> int:
        return self.edge.id

    def set_id(self, new_id: int) -> None:
        self.edge.id = new_id

    def get_ref_id(self) -> int:
        return int(self.edge.model_extra.get("REF_ID") or "-1")

    def get_src_id(self) -> int:
        return self.edge.src_id

    def get_dst_id(self) -> int:
        return self.edge.dst_id

    def set_src_id(self, src_id: int | None) -> None:
        self.edge.src_id = src_id

    def set_dst_id(self, dst_id: int | None) -> None:
        self.edge.dst_id = dst_id

    def get_time(self) -> int:
        return self.edge.time

    def get_op_type(self) -> str:
        return self.edge.optype

    def get_token(self) -> str:
        model = self.edge.model_dump(by_alias=True)
        return "_".join([model["_label"], self.edge.optype])

    def translate_node_ids(self, translation: dict[int, int]) -> None:
        self.set_src_id(translation[self.get_src_id()])
        self.set_dst_id(translation[self.get_dst_id()])

    def __eq__(self, other: "Edge") -> bool:
        return self.get_id() == other.get_id()

    def __hash__(self):
        return hash(self.get_id())

    def to_dot_args(self) -> dict[str, any]:
        # model = self.model_dump(by_alias=True, exclude={'time'})
        args = {"color": "black", "label": ""}
        if self.get_op_type() == "VIRTUAL":
            args["color"] = "blue"
        if self.marked:
            args["color"] = "green"
        # if self.time is not None:
        #     args['label'] += format_timestamp(self.time)
        args["label"] += self.edge.label
        return args

    __json_attributes: dict[str, str] = {
        "EVENT_START": "long",
        "ACCESS_AMOUNT": "long",
        "OPTYPE": "string",
        "PROC_CREATE_INHERIT": "boolean",
        "REL_TIME_START": "long",
        "IS_ALERT": "boolean",
        "TIME_START": "long",
        "REL_TIME_END": "long",
        "ALERT_INFO": "string",
        "EVENT_START_STR": "string",
        "TIME_END": "long",
        "EVENT_END_STR": "string",
        "PATH_NAME": "string",
        "EVENT_END": "long",
        "REF_ID": "long",
        "FT_SOURCE": "boolean"
    }

    def to_json_dict(self) -> dict:
        model = self.edge.model_dump(by_alias=True)
        json_dict = {}
        NOT_FOUND_EDGE = []
        for attribute, value in model.items():
            if attribute not in self.__json_attributes and not attribute.startswith("_"):
                NOT_FOUND_EDGE.add(attribute)
            value_type = self.__json_attributes.get(attribute, "string")
            json_dict[attribute] = json_value(value, value_type)

        json_dict["_id"] = str(self.get_id())
        json_dict["_type"] = "edge"
        json_dict["_outV"] = str(self.get_src_id())
        json_dict["_inV"] = str(self.get_dst_id())
        json_dict["_label"] = str(self.edge.label)
        return json_dict
