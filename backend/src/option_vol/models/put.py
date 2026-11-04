from src.option_vol.models import BaseOption


class Put(BaseOption):
    is_call = False
