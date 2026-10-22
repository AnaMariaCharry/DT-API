
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cqr.pkl"


def cargar_modelo():
    if not MODEL_PATH.is_file():
        return None

    artefacto = joblib.load(MODEL_PATH)

    claves = {
        "modelo_q10_conforme",
        "modelo_q90_conforme",
        "margen_conforme",
        "features",
    }

    if not isinstance(artefacto, dict) or not claves.issubset(artefacto):
        raise ValueError("El archivo CQR no tiene la estructura esperada.")

    return artefacto


def predecir_intervalo(datos: pd.DataFrame, artefacto: dict):
    columnas = artefacto["features"]

    if list(datos.columns) != columnas:
        raise ValueError("Las columnas no coinciden con las del modelo.")

    margen = float(artefacto["margen_conforme"])

    if not np.isfinite(margen):
        raise ValueError("El margen conforme no es válido.")

    inferior = float(
        artefacto["modelo_q10_conforme"].predict(datos)[0] - margen
    )

    superior = float(
        artefacto["modelo_q90_conforme"].predict(datos)[0] + margen
    )

    if not np.isfinite(inferior) or not np.isfinite(superior):
        raise ValueError("El modelo produjo valores no válidos.")

    if inferior > superior:
        raise ValueError("El intervalo CQR tiene límites invertidos.")

    return {
        "limite_inferior_horas": inferior,
        "limite_superior_horas": superior,
    }
