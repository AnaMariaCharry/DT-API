
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TicketEntrada(BaseModel):
    prioridad: Literal["BAJA", "MEDIA", "ALTA", "URGENTE"]
    modulo: str = Field(min_length=1)
    equipo_soporte: str = Field(min_length=1)
    area_encargada: str = Field(min_length=1)
    version_sistema: str = Field(min_length=1)
    fecha_creacion: datetime
