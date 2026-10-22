
from fastapi import FastAPI, HTTPException

from api.schemas import TicketEntrada
from api.preprocessing import preparar_ticket
from api.model_service import cargar_modelo, predecir_intervalo


app = FastAPI(
    title="API de Predicción de Horas - Mesa de Ayuda",
    description="API para estimar intervalos de horas mediante CQR",
    version="1.0.0"
)


@app.get("/")
def inicio():
    return {
        "mensaje": "API de predicción de horas funcionando",
        "estado": "activo"
    }


@app.get("/health")
def verificar_estado():
    modelo = cargar_modelo()

    return {
        "estado": "ok",
        "modelo_cqr": "disponible" if modelo is not None else "pendiente"
    }


@app.post("/predict")
def predecir(ticket: TicketEntrada):
    modelo = cargar_modelo()

    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo CQR todavía no está disponible."
        )

    datos = preparar_ticket(ticket)

    try:
        resultado = predecir_intervalo(datos, modelo)
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    return resultado
