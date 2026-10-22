
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "tickets_mesa_ayuda.xlsx"
SPLIT = ROOT / "data" / "split_temporal.csv"
CLASS_NB = ROOT / "notebooks" / "01_clasificacion_sla.ipynb"


def verificar_archivos():
    archivos = [DATA, SPLIT, CLASS_NB]

    for archivo in archivos:
        if not archivo.is_file():
            raise FileNotFoundError(
                f"No se encontró: {archivo}"
            )

    print("Archivos de entrada verificados correctamente.")


def cargar_datos():
    hojas = pd.read_excel(
        DATA,
        sheet_name=None,
        engine="openpyxl",
        keep_default_na=False,
        na_values=[""]
    )

    if "Tickets" not in hojas:
        raise ValueError("No existe la hoja Tickets.")

    datos = hojas["Tickets"]
    cerrados = datos.loc[datos["ETAPA"].eq("Done")].copy()

    if len(cerrados) != 770:
        raise ValueError(
            f"Se esperaban 770 tickets cerrados, pero hay {len(cerrados)}."
        )

    print("Tickets cerrados:", len(cerrados))

    return cerrados


def dividir_datos(cerrados):
    division = pd.read_csv(SPLIT)
    division = division.sort_values(
        "orden_temporal",
        kind="mergesort"
    )

    if len(division) != 770:
        raise ValueError("La división temporal no tiene 770 tickets.")

    if not division["ID"].is_unique:
        raise ValueError("Hay IDs duplicados en la división.")

    if set(division["ID"]) != set(cerrados["ID"]):
        raise ValueError("Los IDs no coinciden con el Excel.")

    datos = cerrados.set_index("ID", drop=False)

    ids_train = division.loc[
        division["conjunto"].eq("TRAIN"), "ID"
    ].tolist()

    ids_test = division.loc[
        division["conjunto"].eq("TEST"), "ID"
    ].tolist()

    train = datos.loc[ids_train].copy()
    test = datos.loc[ids_test].copy()

    if len(train) != 616 or len(test) != 154:
        raise ValueError("Los tamaños de TRAIN y TEST no coinciden.")

    print("Entrenamiento:", len(train), "tickets")
    print("Prueba:", len(test), "tickets")

    return train, test


def verificar_fechas(train, test):
    columna = "FECHA DE CREACION"

    train[columna] = pd.to_datetime(
        train[columna],
        errors="raise"
    )

    test[columna] = pd.to_datetime(
        test[columna],
        errors="raise"
    )

    if train[columna].isna().any() or test[columna].isna().any():
        raise ValueError("Se encontraron fechas vacías.")

    if not train[columna].is_monotonic_increasing:
        raise ValueError("TRAIN no está ordenado por fecha.")

    if not test[columna].is_monotonic_increasing:
        raise ValueError("TEST no está ordenado por fecha.")

    if not train[columna].max() < test[columna].min():
        raise ValueError(
            "La división temporal no respeta el orden de las fechas."
        )

    print("Orden temporal verificado correctamente.")


def preparar_variables(train, test):
    columnas = [
        "PRIORIDAD",
        "MÓDULO",
        "EQUIPO DE SOPORTE",
        "AREA ENCARGADA",
        "VERSIÓN DEL SISTEMA",
        "dia_semana",
        "mes",
        "hora",
    ]

    for datos in (train, test):
        fecha = datos["FECHA DE CREACION"]

        datos["dia_semana"] = fecha.dt.dayofweek
        datos["mes"] = fecha.dt.month
        datos["hora"] = fecha.dt.hour

    X_train = train[columnas].copy()
    X_test = test[columnas].copy()

    y_train = train["HORAS DEDICADAS"].astype(float).copy()
    y_test = test["HORAS DEDICADAS"].astype(float).copy()

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Hay valores vacíos en las variables.")

    print("Variables del modelo preparadas correctamente.")
    print("Número de variables:", len(columnas))
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    return X_train, X_test, y_train, y_test


def crear_codificador():
    prioridades = ["BAJA", "MEDIA", "ALTA", "URGENTE"]

    columnas_nominales = [
        "MÓDULO",
        "EQUIPO DE SOPORTE",
        "AREA ENCARGADA",
        "VERSIÓN DEL SISTEMA",
    ]

    columnas_temporales = [
        "dia_semana",
        "mes",
        "hora",
    ]

    codificador = ColumnTransformer(
        transformers=[
            (
                "prioridad",
                OrdinalEncoder(
                    categories=[prioridades],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["PRIORIDAD"],
            ),
            (
                "nominales",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                columnas_nominales,
            ),
            (
                "temporales",
                "passthrough",
                columnas_temporales,
            ),
        ],
        verbose_feature_names_out=False,
    )

    print("Codificador CQR creado correctamente.")

    return codificador


def separar_calibracion(X_train, y_train):
    n_calib = round(len(X_train) * 0.2)

    indices_ajuste = X_train.index[:-n_calib]
    indices_calibracion = X_train.index[-n_calib:]

    X_ajuste = X_train.loc[indices_ajuste].copy()
    y_ajuste = y_train.loc[indices_ajuste].copy()

    X_calibracion = X_train.loc[indices_calibracion].copy()
    y_calibracion = y_train.loc[indices_calibracion].copy()

    if len(X_ajuste) != 493 or len(X_calibracion) != 123:
        raise ValueError(
            "La división de ajuste y calibración no coincide."
        )

    if set(indices_ajuste) & set(indices_calibracion):
        raise ValueError(
            "Hay tickets repetidos entre ajuste y calibración."
        )

    print("Ajuste:", len(X_ajuste), "tickets")
    print("Calibración:", len(X_calibracion), "tickets")

    return X_ajuste, y_ajuste, X_calibracion, y_calibracion

    
def entrenar_modelos_cqr(X_ajuste, y_ajuste, codificador):
    modelo_q10 = Pipeline([
        ("codificar", clone(codificador)),
        (
            "modelo",
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.1,
                random_state=42,
            ),
        ),
    ])

    modelo_q90 = Pipeline([
        ("codificar", clone(codificador)),
        (
            "modelo",
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.9,
                random_state=42,
            ),
        ),
    ])

    print("Entrenando modelo Q10...")
    modelo_q10.fit(X_ajuste, y_ajuste)
    print("Modelo Q10 entrenado correctamente.")

    print("Entrenando modelo Q90...")
    modelo_q90.fit(X_ajuste, y_ajuste)
    print("Modelo Q90 entrenado correctamente.")

    return modelo_q10, modelo_q90


def calcular_margen_conforme(
    modelo_q10,
    modelo_q90,
    X_calibracion,
    y_calibracion,
):
    pred_q10 = modelo_q10.predict(X_calibracion)
    pred_q90 = modelo_q90.predict(X_calibracion)

    valores_reales = y_calibracion.to_numpy()

    puntajes = np.maximum(
        pred_q10 - valores_reales,
        valores_reales - pred_q90,
    )

    alpha = 0.10
    n_calib = len(X_calibracion)

    nivel = min(
        1.0,
        np.ceil((n_calib + 1) * (1 - alpha)) / n_calib,
    )

    margen = float(np.quantile(puntajes, nivel))

    print("Nivel de cuantil:", nivel)
    print("Margen conforme (horas):", margen)

    return margen

    
def evaluar_cqr(
    modelo_q10,
    modelo_q90,
    margen_conforme,
    X_test,
    y_test,
):
    limite_inferior = (
        modelo_q10.predict(X_test) - margen_conforme
    )

    limite_superior = (
        modelo_q90.predict(X_test) + margen_conforme
    )

    valores_reales = y_test.to_numpy()

    cobertura = float(
        (
            (valores_reales >= limite_inferior)
            & (valores_reales <= limite_superior)
        ).mean()
    )

    ancho_medio = float(
        (limite_superior - limite_inferior).mean()
    )

    print("Cobertura CQR:", cobertura)
    print("Ancho medio del intervalo (horas):", ancho_medio)

    return cobertura, ancho_medio

    
def guardar_modelo_cqr(
    modelo_q10,
    modelo_q90,
    margen_conforme,
):
    ruta_modelo = ROOT / "models" / "cqr.pkl"

    if ruta_modelo.exists():
        raise FileExistsError(
            "Ya existe cqr.pkl. No se sobrescribirá."
        )

    artefacto = {
        "modelo_q10_conforme": modelo_q10,
        "modelo_q90_conforme": modelo_q90,
        "margen_conforme": margen_conforme,
        "features": [
            "PRIORIDAD",
            "MÓDULO",
            "EQUIPO DE SOPORTE",
            "AREA ENCARGADA",
            "VERSIÓN DEL SISTEMA",
            "dia_semana",
            "mes",
            "hora",
        ],
    }

    joblib.dump(artefacto, ruta_modelo)

    print("Modelo CQR guardado correctamente.")
    print("Ubicación:", ruta_modelo)



if __name__ == "__main__":
    verificar_archivos()
    cerrados = cargar_datos()
    train, test = dividir_datos(cerrados)
    verificar_fechas(train, test)

    X_train, X_test, y_train, y_test = preparar_variables(
        train, test
    )

    codificador = crear_codificador()

    X_ajuste, y_ajuste, X_calibracion, y_calibracion = (
        separar_calibracion(X_train, y_train)
    )

    modelo_q10, modelo_q90 = entrenar_modelos_cqr(
        X_ajuste,
        y_ajuste,
        codificador,
    )

    margen_conforme = calcular_margen_conforme(
        modelo_q10,
        modelo_q90,
        X_calibracion,
        y_calibracion,
    )

    cobertura, ancho_medio = evaluar_cqr(
        modelo_q10,
        modelo_q90,
        margen_conforme,
        X_test,
        y_test,
    )

    guardar_modelo_cqr(
        modelo_q10,
        modelo_q90,
        margen_conforme,
    )
