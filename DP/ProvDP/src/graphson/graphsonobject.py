from pydantic import BaseModel, root_validator

from .utility import string_to_field


# noinspection PyMethodParameters
class GraphsonObject(BaseModel):
    # noinspection PyDeprecation
    @root_validator(pre=True)
    def extract_values(cls, values):
        result = {}
        for key, value in values.items():
            if isinstance(value, list):
                assert len(value) == 1
                value = value[0]
            if key.startswith("_") or not isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value["value"]
        return result

    class Config:
        extra = "allow"

    def to_dict(self):
        new_dict = {}
        model = self.model_dump(by_alias=True)
        for key, value in model.items():
            str_value = str(value)
            if key.startswith("_"):
                new_dict[key] = str_value
            else:
                new_dict[key] = string_to_field(str_value)
        return new_dict
