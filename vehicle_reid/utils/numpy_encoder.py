import json
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Extends the JSONEncoder with a simple checker and converter for numpy arrays."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
