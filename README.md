# API de predicción de horas mediante CQR

## Descripción

Proyecto de tesis que estima intervalos de horas dedicadas a tickets de mesa de
ayuda mediante Conformalized Quantile Regression (CQR). Dos modelos estiman Q10
y Q90 y un margen conforme ajusta los extremos. Las horas representan esfuerzo
registrado, no necesariamente tiempo calendario hasta resolver el ticket.

## Tecnologías utilizadas

Python 3.11, FastAPI, Uvicorn, Pydantic, scikit-learn, pandas, NumPy, SciPy, joblib
y pytest. Las pruebas HTTP utilizan httpx y TestClient. El script académico usa
openpyxl para leer Excel.

## Estructura del proyecto

```text
api/                     Endpoints, validación, variables e inferencia
models/cqr.pkl           Pipelines entrenados Q10/Q90 y margen conforme
models/entrenar_cqr.py    Script académico; no ejecutar para iniciar la API
models/export_cqr.py      Exportación desde los objetos del notebook original
tests/                   Pruebas automatizadas de la API y el modelo
requirements.txt         Dependencias de ejecución y pruebas
```

Datos, notebooks, resultados y plan académico permanecen en el proyecto local y
se excluyen de Git hasta confirmar autorización para publicarlos. No se necesitan
para ejecutar la API ni sus pruebas. Esta distribución reproduce la inferencia
con el modelo guardado; el análisis académico requiere los insumos por separado.

## Requisitos

Python 3.11 de 64 bits y Git. Se recomienda Python 3.11.6, usado en las pruebas.
Conservar scikit-learn 1.5.2 para cargar el modelo serializado. `requirements.txt`
está destinado a la API y sus pruebas, no al conjunto completo de notebooks.

## Instalación

Desde PowerShell:

```powershell
git clone https://github.com/AnaMariaCharry/DT-API.git
cd DT-API
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell impide activar el entorno, usar directamente su intérprete:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

## Ejecución de la API

Desde la raíz, con el entorno activado:

```powershell
python -m uvicorn api.main:app --reload
```

Estas URL son locales y requieren que la API esté ejecutándose:

- http://127.0.0.1:8000/ — respuesta inicial.
- http://127.0.0.1:8000/health — disponibilidad del modelo.
- http://127.0.0.1:8000/docs — Swagger y pruebas interactivas.
- http://127.0.0.1:8000/redoc — documentación de endpoints.

## Ejemplo de predicción

En Swagger, abrir `POST /predict`, seleccionar **Try it out** y enviar:

```json
{
  "prioridad": "ALTA",
  "modulo": "Otro",
  "equipo_soporte": "Soporte en Desarrollo a la medida",
  "area_encargada": "Soporte Desarrollo",
  "version_sistema": "19",
  "fecha_creacion": "2026-05-15T08:38:04.224"
}
```

La API calcula día de semana, mes y hora y devuelve `limite_inferior_horas` y
`limite_superior_horas`. El ejemplo devuelve aproximadamente 4.67493 y 13.47737
horas con el modelo guardado. Las prioridades válidas son BAJA, MEDIA, ALTA y
URGENTE. Entradas inválidas responden 422. Si falta el modelo, `/predict`
responde 503 y `/health` informa `pendiente`.

## Pruebas automatizadas

```powershell
python -m pytest tests -v
```

La suite contiene 21 casos: entradas válidas e inválidas, prioridades, salud,
ausencia simulada del modelo, variables temporales, límites numéricos, paridad con
los pipelines guardados e integridad de archivos. No entrena ni ejecuta notebooks.
`tests/conftest.py` permite importar el paquete local. El comando funciona con
un entorno Python normal y no requiere `.venv/runtime`.

## Resultados del modelo

- 770 tickets cerrados utilizados.
- 616 registros para entrenamiento y calibración: 493 de ajuste y 123 de calibración.
- 154 registros de prueba.
- Cobertura observada en el notebook: aproximadamente 88.31 %.
- Ancho promedio del intervalo en el notebook: aproximadamente 7.56 horas.

Son resultados documentados en la tesis; no se recalcularon para esta publicación.
La cobertura en TEST no garantiza precisión para un ticket individual. El Excel
local identifica los datos como sintéticos. No se añaden resultados nuevos.

## Modelo y datos

`models/cqr.pkl` contiene los pipelines completos, categorías aprendidas, margen
conforme y orden de las ocho variables. No contiene el Excel ni los campos de
clientes, responsables o textos de tickets. Ocupa aproximadamente 426 KiB.
Solo cargar artefactos de confianza: joblib/pickle puede ejecutar código al
deserializar. No es necesario reentrenarlo para usar la API.

Los datasets y resultados por ticket no se distribuyen. Para reproducir el
entrenamiento, solicitar a la autora los insumos y su autorización. No ejecutar
`models/entrenar_cqr.py` como paso de instalación.
