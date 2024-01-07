from typing import List

from fastapi_utils.api_model import APIModel


class Surface(APIModel):
    x: List
    y: List
    z: List[List[float]]
