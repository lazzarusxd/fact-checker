import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .database import Database
from .orchestrator import Orchestrator

_state: dict = {}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db = Database()
    await db.connect()
    orch = Orchestrator(db)
    print("[startup] carregando modelo de embeddings e indexando corpus...")
    orch.warmup_sync()
    await orch.load_learned()
    _state["db"] = db
    _state["orch"] = orch
    print("[startup] pronto.")
    yield
    await db.close()


app = FastAPI(title="Verificador Factual Multiagente", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/verify")
async def verify(request: Request):
    body = await request.json()
    claim = (body.get("claim") or "").strip()
    if not claim:
        return JSONResponse({"error": "Informe uma alegação."}, status_code=400)
    try:
        result = await _state["orch"].verify(claim)
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
