import sys

from fastapi import FastAPI
from fastapi_utils.inferring_router import InferringRouter
from starlette.middleware.cors import CORSMiddleware

from src.api.explorer.controller import router as explorer_router

sys.path.append("..")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = InferringRouter()
router.include_router(explorer_router)

app.include_router(router)
from src.core.db import create_tables

create_tables()


@app.get("/")
def home():
    return "Hello, World!"
