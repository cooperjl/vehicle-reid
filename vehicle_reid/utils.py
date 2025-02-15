import json
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def pad_label(label: int, dataset: str) -> str:
    match dataset:
        case "veri":
            # less than 1000 classes, pad to 3 digits
            return f"{label:03}"
        case "vric":
            # pad to 4 digits
            return f"{label:04}"
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

