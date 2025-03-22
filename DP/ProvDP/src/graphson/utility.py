from datetime import datetime


def format_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(int(timestamp / 1e9)).strftime("%Y-%m-%d %H:%M:%S")


def string_to_field(value: str):
    try:
        return {"type": "long", "value": int(value)}
    except ValueError:
        return {"type": "string", "value": value}
