from datetime import date

import numpy as np
import sqlalchemy as sa
from fastapi_utils.api_model import APIModel
from pydantic import validator
from sqlalchemy.orm import Session

from src.core.db import Base
from src.option_vol.models import BaseOption, Environment


class OptionORM(Base):
    __tablename__ = "option"

    name = sa.Column(sa.String, primary_key=True, nullable=False)
    type = sa.Column(sa.String, nullable=False)
    underlying = sa.Column(sa.String, nullable=False)
    strike = sa.Column(sa.Integer, nullable=False)
    maturity = sa.Column(sa.Date, nullable=False)
    price = sa.Column(sa.Float)
    delta = sa.Column(sa.Float)
    gamma = sa.Column(sa.Float)
    vega = sa.Column(sa.Float)
    theta = sa.Column(sa.Float)
    rho = sa.Column(sa.Float)
    implied_vol = sa.Column(sa.Float)

    @classmethod
    def get_by_underlying(cls, session: Session, underlying: str):
        return session.query(cls).filter(cls.underlying == underlying)

    @classmethod
    def get_by_underlying_and_type(cls, session: Session, underlying: str, option_type: str):
        return cls.get_by_underlying(session, underlying).filter(cls.type == option_type)

    @classmethod
    def get_options_near_spot(cls, session: Session, underlying: str, option_type: str, depth: int):
        relative_strike = cls.strike / Environment().get_spot(underlying)
        return cls.get_by_underlying_and_type(session, underlying, option_type) \
            .filter(relative_strike <= 1 + depth / 100).filter(1 - depth / 100 <= relative_strike)

    @classmethod
    def remove_by_underlying(cls, session: Session, underlying: str):
        return session.query(cls).filter(cls.underlying == underlying).delete()

    @staticmethod
    def from_option(option: BaseOption):
        return OptionORM(**{k: v for k, v in option.__dict__.items() if k in OptionORM.__dict__})


class Option(APIModel):
    name: str
    type: str
    underlying: str
    strike: int
    maturity: date
    price: float = 0
    delta: float = 0
    gamma: float = 0
    vega: float = 0
    theta: float = 0
    rho: float = 0
    implied_vol: float = 0

    @validator('delta', 'gamma', 'vega', 'theta', 'rho', 'implied_vol', pre=True)
    def default(cls, v: float) -> float:
        return 0. if v in (np.inf, np.nan) else v

    @validator('delta', 'gamma', 'vega', 'theta', 'rho', 'implied_vol')
    def round(cls, v):
        return round(v, 5)
