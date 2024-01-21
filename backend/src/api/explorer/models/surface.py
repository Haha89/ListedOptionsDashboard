import datetime
from typing import List, Optional

from fastapi_utils.api_model import APIModel
from pydantic import BaseModel


class SurfaceRequest(BaseModel):
    ticker: str
    option_type: str
    field: str
    range: int
    maturity: Optional[datetime.date] = None


class Surface(APIModel):
    x: List
    y: List
    z: List[List[float]]
