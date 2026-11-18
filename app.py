"""Servicio JSON NFCorpus. Se genera desde el notebook."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt

from motor_ri import buscar


# FastAPI expone estas rutas y genera /docs y /openapi.json.
app = FastAPI(
    title="Sistema de recuperación NFCorpus",
    description="Búsqueda de documentos originales con TF-IDF o BM25.",
    version="1.0.0",
)
RUTA_INDICE = Path(__file__).with_name("indice_nfc.joblib")


class EstadoServicio(BaseModel):
    # Define la forma de la respuesta de GET /health.
    status: str
    documents: int


class SolicitudBusqueda(BaseModel):
    # Los tipos y Field se reflejan en la documentación y validan el JSON.
    query: str = Field(min_length=1, description="Consulta de texto libre")
    model: Literal["bm25", "tfidf"]
    top_k: StrictInt = Field(default=10, gt=0, description="Número de resultados")


class ResultadoBusqueda(BaseModel):
    # Estos son los cinco campos exigidos para cada documento recuperado.
    rank: int
    document_id: str
    score: float
    title: str
    snippet: str


class RespuestaBusqueda(BaseModel):
    model: Literal["bm25", "tfidf"]
    results: list[ResultadoBusqueda]


@lru_cache(maxsize=1)
def cargar_sistema():
    # Una instancia carga el índice guardado solo en su primera petición.
    return joblib.load(RUTA_INDICE)


@app.get("/health", response_model=EstadoServicio)
def health():
    # Cargarlo aquí comprueba que el servicio dispone del índice completo.
    sistema = cargar_sistema()
    return {"status": "ok", "documents": len(sistema["documentos"])}


@app.post("/search", response_model=RespuestaBusqueda)
def search(datos: SolicitudBusqueda):
    # Field rechaza un texto vacío; strip detecta también solo espacios.
    if not datos.query.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "query debe ser texto no vacío."},
        )

    # El motor recibe consulta, modelo y top_k después de validar el JSON.
    sistema = cargar_sistema()
    resultados = buscar(sistema, datos.query, datos.model, datos.top_k)
    return {"model": datos.model, "results": resultados}
