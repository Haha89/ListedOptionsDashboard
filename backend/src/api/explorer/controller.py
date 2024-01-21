import datetime
from typing import List

from fastapi import APIRouter, Depends
from fastapi_utils.cbv import cbv
from sqlalchemy.orm import Session

from src.core.db import get_db
from .models.option import Option
from .models.surface import Surface, SurfaceRequest
from .service import ExplorerService

router = APIRouter()


@cbv(router)
class ExplorerController:
    session: Session = Depends(get_db)

    @router.get("/options/{ticker}", response_model=List[Option])
    def read_options(self, ticker, refresh: bool = False, maturity: datetime.date = None) -> List[Option]:
        return [Option.from_orm(o) for o in ExplorerService.get_options(self.session, ticker, refresh, maturity)]

    @router.post("/surface", response_model=Surface)
    def get_surface_p(self, request: SurfaceRequest) -> Surface:
        return ExplorerService.get_surface(self.session, request)

    @router.get("/spot/{ticker}")
    def get_spot(self, ticker) -> float:
        from src.option_vol.models import Environment
        return Environment().get_spot(underlying=ticker)
