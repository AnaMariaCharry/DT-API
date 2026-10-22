"""Pruebas de integración con el CQR guardado, sin entrenamiento."""
import hashlib
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import model_service
from api.main import app
from api.preprocessing import preparar_ticket
from api.schemas import TicketEntrada

ROOT = Path(__file__).resolve().parents[1]
TICKET = {
    'prioridad': 'ALTA', 'modulo': 'Otro',
    'equipo_soporte': 'Soporte en Desarrollo a la medida',
    'area_encargada': 'Soporte Desarrollo', 'version_sistema': '19',
    'fecha_creacion': '2026-05-15T08:38:04.224',
}
FEATURES = ['PRIORIDAD', 'MÓDULO', 'EQUIPO DE SOPORTE', 'AREA ENCARGADA',
            'VERSIÓN DEL SISTEMA', 'dia_semana', 'mes', 'hora']


def snapshot():
    paths = [ROOT / 'models/cqr.pkl', ROOT / 'data/split_temporal.csv']
    for folder in ('notebooks', 'data', 'outputs', 'api', 'models'):
        paths.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(p for p in ROOT.iterdir() if p.is_file())
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in set(paths) if p.is_file()}


@pytest.fixture(scope='session', autouse=True)
def originales_intactos():
    before = snapshot()
    yield
    assert snapshot() == before, 'Una prueba modificó archivos originales'


@pytest.fixture
def client():
    with TestClient(app) as connection:
        yield connection


def test_prediccion_valida(client):
    response = client.post('/predict', json=TICKET)
    assert response.status_code == 200, response.text
    assert {'limite_inferior_horas', 'limite_superior_horas'} <= response.json().keys()


@pytest.mark.parametrize('campo,valor', [
    ('prioridad', 'SUPERALTA'), ('prioridad', ''), ('prioridad', 123),
    ('fecha_creacion', 'fecha-invalida'), ('modulo', ''),
    ('equipo_soporte', ''), ('area_encargada', ''), ('version_sistema', ''),
])
def test_entrada_invalida(client, campo, valor):
    response = client.post('/predict', json={**TICKET, campo: valor})
    assert response.status_code == 422, response.text
    assert any(campo in error['loc'] for error in response.json()['detail'])


def test_campo_obligatorio_faltante(client):
    payload = TICKET.copy()
    del payload['modulo']
    response = client.post('/predict', json=payload)
    assert response.status_code == 422
    assert any('modulo' in error['loc'] for error in response.json()['detail'])


@pytest.mark.parametrize('prioridad', ['BAJA', 'MEDIA', 'ALTA', 'URGENTE'])
def test_prioridades_permitidas(client, prioridad):
    response = client.post('/predict', json={**TICKET, 'prioridad': prioridad})
    assert response.status_code == 200, response.text


def test_health_modelo_disponible(client):
    assert model_service.MODEL_PATH.is_file()
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'estado': 'ok', 'modelo_cqr': 'disponible'}


def test_limites_numericos_finitos_ordenados(client):
    response = client.post('/predict', json=TICKET)
    assert response.status_code == 200, response.text
    data = response.json()
    lower, upper = data['limite_inferior_horas'], data['limite_superior_horas']
    assert type(lower) in (int, float) and type(upper) in (int, float)
    assert math.isfinite(lower) and math.isfinite(upper)
    assert lower <= upper


@pytest.mark.parametrize('endpoint', ['/health', '/predict'])
def test_modelo_ausente(client, monkeypatch, endpoint):
    original = model_service.MODEL_PATH
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    absent = ROOT / 'models' / '__modelo_inexistente_para_prueba__.pkl'
    assert not absent.exists()
    monkeypatch.setattr(model_service, 'MODEL_PATH', absent)
    assert model_service.cargar_modelo() is None
    if endpoint == '/health':
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json()['modelo_cqr'] == 'pendiente'
    else:
        response = client.post(endpoint, json=TICKET)
        assert response.status_code == 503
        assert response.json()['detail']
    assert original.is_file()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest


def test_preprocesamiento_ticket_711():
    frame = preparar_ticket(TicketEntrada(**TICKET))
    assert list(frame.columns) == FEATURES
    assert frame.iloc[0].tolist() == [
        'ALTA', 'Otro', 'Soporte en Desarrollo a la medida',
        'Soporte Desarrollo', '19', 4, 5, 8]


def test_paridad_con_modelo_guardado(client):
    # Referencia independiente del preprocesamiento y del servicio de la API.
    artifact = joblib.load(ROOT / 'models/cqr.pkl')
    assert artifact['features'] == FEATURES
    frame = pd.DataFrame([[
        'ALTA', 'Otro', 'Soporte en Desarrollo a la medida',
        'Soporte Desarrollo', '19', 4, 5, 8]], columns=FEATURES)
    margin = float(artifact['margen_conforme'])
    expected = [float(artifact['modelo_q10_conforme'].predict(frame)[0] - margin),
                float(artifact['modelo_q90_conforme'].predict(frame)[0] + margin)]
    response = client.post('/predict', json=TICKET)
    assert response.status_code == 200, response.text
    data = response.json()
    np.testing.assert_allclose(
        [data['limite_inferior_horas'], data['limite_superior_horas']],
        expected, rtol=1e-12, atol=1e-12)
    print(f'Ticket 711: inferior={expected[0]:.12f}, superior={expected[1]:.12f}; real=9.9')


def test_prediccion_no_modifica_archivos(client):
    before = snapshot()
    response = client.post('/predict', json=TICKET)
    assert response.status_code == 200, response.text
    assert snapshot() == before
