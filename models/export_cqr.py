"""Exportación de los objetos originales de la sección 36.2b, sin reentrenar."""
from copy import copy
from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer


def exportar_cqr(ns):
    root = Path(ns['ROOT']).resolve()
    assert (root / 'notebooks/02_regresion_horas.ipynb').is_file()
    columns = list(ns['features'])
    X = ns['X_test'].loc[:, columns]
    pipelines = {}
    for name in ('modelo_q10_conforme', 'modelo_q90_conforme'):
        original = ns[name]
        # TrainOnlyEncoder solo sobreescribe fit_transform para auditar TRAIN.
        # Conservamos todos los atributos ajustados, sin sus globals del notebook.
        trained = original.named_steps['codificar']
        standard = ColumnTransformer([])
        standard.__dict__ = trained.__dict__.copy()
        np.testing.assert_array_equal(trained.transform(X), standard.transform(X))
        portable = copy(original)
        portable.steps = [('codificar', standard), ('modelo', original.named_steps['modelo'])]
        np.testing.assert_allclose(portable.predict(X), original.predict(X), rtol=0, atol=0)
        assert list(portable.feature_names_in_) == columns
        pipelines[name] = portable

    margin = float(ns['margen_conforme'])
    assert np.isfinite(margin)
    assert set(ns['idx_ajuste']).isdisjoint(ns['idx_calibracion'])
    assert set(ns['idx_ajuste']).isdisjoint(X.index)
    assert set(ns['idx_calibracion']).isdisjoint(X.index)
    expected = np.column_stack([
        ns['pred_q10_test_conforme'], ns['pred_q90_test_conforme']])
    actual = np.column_stack([
        pipelines['modelo_q10_conforme'].predict(X) - margin,
        pipelines['modelo_q90_conforme'].predict(X) + margin])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    artifact = {
        **pipelines,
        'margen_conforme': margin,
        'features': columns,
        'priority_order': list(ns['priority_order']),
        'preprocesamiento': {k: p.named_steps['codificar'] for k, p in pipelines.items()},
        'transformaciones_temporales': {
            'origen': 'FECHA DE CREACION', 'conversion': "pd.to_datetime(errors='raise')",
            'dia_semana': 'dt.dayofweek (lunes=0)', 'mes': 'dt.month (1..12)',
            'hora': 'dt.hour (0..23)', 'zona_horaria': 'sin conversión; igual al notebook'},
        'metadatos': {
            'version_artefacto': '1.0.0', 'cuantiles': [0.1, 0.9],
            'alpha_original': float(ns['alpha']), 'nivel_calibracion': float(ns['nivel']),
            'objetivo_texto_notebook': 0.80,
            'cobertura_test': float(ns['cobertura_conforme']),
            'ancho_medio_test': float(ns['ancho_medio_conforme']),
            'n_ajuste': len(ns['idx_ajuste']), 'n_calibracion': len(ns['idx_calibracion']),
            'n_test': len(X), 'random_state': ns['RANDOM_STATE'],
            'intervalo': '[Q10(X) - margen_conforme, Q90(X) + margen_conforme]',
            'recorte_limites': None, 'escala_target': 'horas, sin logaritmo',
            'python': sys.version,
            'dependencias': {p: importlib.metadata.version(p)
                             for p in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib')},
        },
    }
    folder = root / 'models'
    folder.mkdir(exist_ok=True)
    path = folder / 'cqr.pkl'
    joblib.dump(artifact, path)
    assert path.is_file() and path.stat().st_size > 0
    # El hijo carga exclusivamente el archivo, sin importar esta función ni el notebook.
    # La muestra y las referencias viajan por stdin, sin archivos auxiliares.
    payload = json.dumps({'X': X.to_json(orient='split'), 'expected': expected.tolist(),
                          'margin': margin, 'features': columns})
    verifier = '''
import io, json, sys
import joblib, numpy as np, pandas as pd
a = joblib.load(sys.argv[1])
t = json.load(sys.stdin)
X = pd.read_json(io.StringIO(t['X']), orient='split')
assert a['features'] == t['features'] == list(X.columns)
assert a['margen_conforme'] == t['margin']
q10 = a['modelo_q10_conforme'].predict(X)
q90 = a['modelo_q90_conforme'].predict(X)
np.testing.assert_allclose(np.column_stack([q10-a['margen_conforme'],
    q90+a['margen_conforme']]), t['expected'], rtol=1e-12, atol=1e-12)
assert len(a['modelo_q10_conforme'].predict(X.iloc[:1])) == 1
assert len(a['modelo_q90_conforme'].predict(X.iloc[:1])) == 1
print('OK: carga independiente, columnas, margen, registro y paridad en todo TEST')
'''
    result = subprocess.run([sys.executable, '-c', verifier, str(path)], input=payload,
                            text=True, capture_output=True, cwd=root, check=True)
    print(result.stdout.strip())
    print(f'Artefacto exportado: {path} ({path.stat().st_size} bytes)')
    return artifact
