"""Permite importar el paquete local sin instalar el proyecto."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
