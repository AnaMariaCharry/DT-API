# Modelo CQR

`cqr.pkl` contiene `modelo_q10_conforme`, `modelo_q90_conforme`,
`margen_conforme` y `features`. Los dos modelos son pipelines completos de
scikit-learn: codificación ajustada y HistGradientBoostingRegressor.
Se carga mediante joblib con scikit-learn 1.5.2, Python 3.11 y las dependencias
indicadas en el requirements.txt de la raíz.

Q10 y Q90 representan los percentiles 10 y 90. El intervalo se calcula restando
el margen conforme a Q10 y sumándolo a Q90, sin recortar los límites.
La API deriva día de semana (lunes=0), mes y hora a partir de la fecha de creación.
Las ocho columnas están definidas en `api/preprocessing.py` y en `features`.

El artefacto existente utiliza ColumnTransformer estándar y no necesita importar
TrainOnlyEncoder ni abrir un notebook. No se ha reentrenado ni modificado para
publicarlo. Cargar únicamente archivos de confianza: joblib/pickle puede ejecutar
código al deserializar.

`entrenar_cqr.py` es el script académico y requiere los datos y el notebook de
clasificación locales, que no se distribuyen hasta confirmar autorización.
`export_cqr.py` corresponde al flujo de exportación desde las variables del
notebook. Ninguno debe ejecutarse para iniciar la API. No sobrescribir el modelo.

Consultar el README de la raíz para instalación, ejecución y pruebas.
