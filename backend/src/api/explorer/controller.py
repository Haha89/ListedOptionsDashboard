from typing import List

from fastapi import APIRouter, Depends
from fastapi_utils.cbv import cbv
from sqlalchemy.orm import Session

from src.core.db import get_db
from .models.option import Option
from .models.surface import Surface
from .service import ExplorerService

router = APIRouter()


@cbv(router)
class ExplorerController:
    session: Session = Depends(get_db)

    @router.get("/options/{ticker}")
    def read_options(self, ticker, refresh: bool = False) -> List[Option]:
        return [Option.from_orm(o) for o in ExplorerService().get_options(self.session, ticker, refresh)]

    @router.get("/surface/{ticker}/{option_type}/{field}/{range}")
    def get_surface(self, ticker: str, option_type: str, field: str, range:int) -> Surface:
        return ExplorerService().get_surface(self.session, ticker, option_type, field, range)

    @router.get("/spot/{ticker}")
    def get_spot(self, ticker) -> float:
        from src.option_vol.models import Environment
        return Environment().get_spot(underlying=ticker)
