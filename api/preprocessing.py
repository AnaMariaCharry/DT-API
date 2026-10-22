
import pandas as pd

from api.schemas import TicketEntrada


FEATURES = [
    "PRIORIDAD",
    "MÓDULO",
    "EQUIPO DE SOPORTE",
    "AREA ENCARGADA",
    "VERSIÓN DEL SISTEMA",
    "dia_semana",
    "mes",
    "hora",
]


def preparar_ticket(ticket: TicketEntrada) -> pd.DataFrame:
    fecha = pd.Timestamp(ticket.fecha_creacion)

    datos = {
        "PRIORIDAD": ticket.prioridad,
        "MÓDULO": ticket.modulo,
        "EQUIPO DE SOPORTE": ticket.equipo_soporte,
        "AREA ENCARGADA": ticket.area_encargada,
        "VERSIÓN DEL SISTEMA": ticket.version_sistema,
        "dia_semana": fecha.dayofweek,
        "mes": fecha.month,
        "hora": fecha.hour,
    }

    return pd.DataFrame([datos], columns=FEATURES)
