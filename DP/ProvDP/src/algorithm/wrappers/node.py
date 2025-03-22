from ..utility import json_value
from ...graphson import NodeType, RawNode


class Node:
    node: RawNode
    incoming_edges: set[int]
    outgoing_edges: set[int]
    marked: bool = False
    min_time: int | None = None
    max_time: int | None = None

    def __init__(self, node: RawNode):
        self.node = node
        self.incoming_edges, self.outgoing_edges = set(), set()

    # Adapter functions (reach into RawNode object)
    def get_id(self) -> int:
        return self.node.id

    def set_id(self, new_id: int) -> None:
        self.node.id = new_id

    def get_type(self) -> NodeType:
        return self.node.type

    # Comparison functions
    def __eq__(self, other: "Node") -> bool:
        return self.get_id() == other.get_id()

    def __hash__(self) -> int:
        return hash(self.get_id())

    # Exporter functions
    def get_token(self) -> str:
        model = self.node.model_dump()
        token = f"{self.node.type}_"
        if self.node.type == NodeType.PROCESS_LET:
            token += model.get("EXE_NAME")
        elif self.node.type == NodeType.FILE:
            if "FILENAME_SET" in model:
                token += model.get("FILENAME_SET")[0]["value"].replace(" ", "_")
        elif self.node.type == NodeType.IP_CHANNEL:
            src_ip = model.get("LOCAL_INET_ADDR")
            dst_ip = model.get("REMOTE_INET_ADDR")
            token += f"{src_ip}_{dst_ip}"
        elif self.node.type == NodeType.VIRTUAL:
            token = "."
        else:
            raise ValueError(f"Unknown node type: {self.node.type}")

        return token

    @staticmethod
    def __format_label(model: dict, label_key_list: list[tuple[str, str]]) -> str:
        return " ".join([f"{label}: {model.get(key)}" for label, key in label_key_list])

    @staticmethod
    def __sanitize(label: str) -> str:
        # Replace backslashes with forward slashes and double quotes with single quotes
        return label.replace("\\", "/").replace('\\"', "'")

    def to_dot_args(self) -> dict[str, any]:
        model = self.node.model_dump(by_alias=True, exclude={"time"})
        match self.get_type():
            case NodeType.PROCESS_LET:
                args = {
                    "color": "black",
                    "shape": "box",
                    "style": "solid",
                    "label": Node.__format_label(
                        model, [("exe_name", "EXE_NAME"), ("cmd", "CMD")]
                    ),
                }
            case NodeType.FILE:
                filename = "no file name"
                if model.get("FILENAME_SET") is not None:
                    filename = model["FILENAME_SET"][0]["value"]
                args = {
                    "color": "pink",
                    "shape": "oval",
                    "style": "filled",
                    "label": "path: " + filename,
                }
            case NodeType.IP_CHANNEL:
                args = {
                    "color": "yellow",
                    "shape": "box",
                    "style": "filled",
                    "label": Node.__format_label(
                        model,
                        [
                            ("srcIP", "LOCAL_INET_ADDR"),
                            ("dstIP", "REMOTE_INET_ADDR"),
                            ("type", "CHANEL_STATE"),
                        ],
                    ),
                }
            case NodeType.VIRTUAL:
                args = {
                    "color": "blue",
                    "shape": "oval",
                    "style": "solid",
                    "label": "VIRTUAL",
                }
            case _:
                raise ValueError(f"Unknown node type: {self.get_type()}")
        if self.marked:
            args["color"] = "greenyellow"
        return {key: Node.__sanitize(value) for key, value in args.items()}

    __json_attributes: dict[str, str] = {
        "REF_DB": "string",
        "DATA_ID": "string",
        "AGENT_ID": "long",
        "FILENAME_SET": "list",
        "FILE_OWNER_GROUP_ID": "string",
        "RENAME_SET": "list",
        "VOL_ID": "string",
        "FILE_OWNER_UID": "string",
        "TYPE": "string",
        "REF_ID": "long",
        "BT_HOPCOUNT": "integer",
        "postgres": "long",
        "_id": "string",
        "_type": "string",
        "PROC_ORDINAL": "string",
        "PID": "long",
        "EXE_NAME": "string",
        "FT_HOPCOUNT": "integer",
        "PROC_STARTTIME": "long",
        "OWNER_GROUP_ID": "string",
        "CMD": "string",
        "OWNER_UID": "string",
        "IS_INCOMING": "boolean",
        "CONN_TYPE": "integer",
        "REMOTE_INET_ADDR": "string",
        "REMOTE_PORT": "integer",
        "LOCAL_PORT": "integer",
        "LOCAL_INET_ADDR": "string",
        "CHANNEL_STATE": "string",
        "CHANNEL_TYPE": "string",
        "PROC_OWNER_UID": "string",
        "PROC_OWNER_GROUP_ID": "string",
    }

    def to_json_dict(self) -> dict:
        model = self.node.model_dump(by_alias=True)
        json_dict = {}
        for attribute, value in model.items():
            value_type = self.__json_attributes.get(attribute, "string")
            json_dict[attribute] = json_value(value, value_type)

        json_dict["_id"] = str(self.get_id())
        json_dict["_type"] = "vertex"
        json_dict["TYPE"] = json_value(self.get_type().value, "string")
        return json_dict
