import cv2
import numpy as np
from collections import defaultdict
import time
from app.utils import procesar_imagen

import json
import os

# Cargar etiquetas reales desde archivo
ETIQUETAS_REALES = {}
try:
    with open('imagenes/etiquetas.json', 'r') as f:
        ETIQUETAS_REALES = json.load(f)
except Exception as e:
    print("No se pudieron cargar las etiquetas reales:", e)


if not ETIQUETAS_REALES:
    print("⚠️ Advertencia: No se encontraron etiquetas reales. La precisión específica no se podrá calcular.")


# Esta clase es útil para acumular resultados por usuario
class ClienteMonitoreo:
    def __init__(self):
        self.data = defaultdict(lambda: {
            "total": 0,
            "buenos": 0,
            "fisura": 0,
            "rotura": 0,
            "tiempos_fisura": [],
            "start_time": time.time()
        })

    # Requiere: predicciones (lista de clases) y frame_name (nombre del archivo)
    def procesar_resultados(self, usuario_id, predicciones, frame_name=None):
        datos = self.data[usuario_id]
        datos["total"] += len(predicciones)

        for clase in predicciones:
            if clase == "bueno":
                datos["buenos"] += 1
            elif clase == "fisura":
                datos["fisura"] += 1
                datos["tiempos_fisura"].append(time.time() - datos["start_time"])
            elif clase == "rotura":
                datos["rotura"] += 1

        # Calcular TP y FP para fisura
        if frame_name:
            nombre_archivo = os.path.basename(frame_name)
            if nombre_archivo in ETIQUETAS_REALES:
                etiqueta_real = ETIQUETAS_REALES[nombre_archivo]
                print(f"[DEBUG] Etiqueta real encontrada para {nombre_archivo}: {etiqueta_real}")
                predice_fisura = "fisura" in predicciones

                if "tp_fisura" not in datos:
                    datos["tp_fisura"] = 0
                    datos["fp_fisura"] = 0

                if predice_fisura and etiqueta_real == "fisura":
                    datos["tp_fisura"] += 1
                elif predice_fisura and etiqueta_real != "fisura":
                    datos["fp_fisura"] += 1
                print(f"[DEBUG {frame_name}] TP={datos.get('tp_fisura',0)} FP={datos.get('fp_fisura',0)} Precision fisura: {round((datos.get('tp_fisura',0) / (datos.get('tp_fisura',0) + datos.get('fp_fisura',0))) * 100, 2) if (datos.get('tp_fisura',0) + datos.get('fp_fisura',0)) > 0 else 0}%")



    def obtener_métricas_finales(self, usuario_id):
        datos = self.data[usuario_id]
        malos = datos["fisura"] + datos["rotura"]
        #precision = round((datos["buenos"] / datos["total"]) * 100, 2) if datos["total"] > 0 else 0
        precision = round((malos / datos["total"]) * 100, 2) if datos["total"] > 0 else 0
        #promedio_fisura = round(sum(datos["tiempos_fisura"]) / len(datos["tiempos_fisura"]), 2) if datos["tiempos_fisura"] else 0
        #--------------------------------------------------
        if datos["tiempos_fisura"]:
            print(f"[DEBUG - PROMEDIO FISURA] Calculando promedio de {len(datos['tiempos_fisura'])} valores.")
            print(f"[DEBUG - PROMEDIO FISURA] tiempos_fisura = {datos['tiempos_fisura']}")
            promedio_fisura = round(sum(datos["tiempos_fisura"]) / len(datos["tiempos_fisura"]), 2)
            print(f"[DEBUG - PROMEDIO FISURA] Resultado: {promedio_fisura:.2f} segundos")
        else:
            print("[DEBUG - PROMEDIO FISURA] No hay tiempos registrados. Promedio = 0")
            promedio_fisura = 0
        
        #--------------------------------------------------
        tp = datos.get("tp_fisura", 0)
        fp = datos.get("fp_fisura", 0)
        precision_fisura = round((tp / (tp + fp)) * 100, 2) if (tp + fp) > 0 else 0
        
        return {
            "total": datos["total"],
            "buenos": datos["buenos"],
            "malos": malos,
            "precision": precision,
            "tiempo_promedio_fisura": promedio_fisura,
            "fisura": datos.get("fisura", 0),
            "rotura": datos.get("rotura", 0),
            "precision_fisura": precision_fisura
        }



    def resetear(self, usuario_id):
        if usuario_id in self.data:
            del self.data[usuario_id]


# Instancia global
cliente_monitoreo = ClienteMonitoreo()

# ✅ Esta es la función que debes importar
def procesar_frame_yolo(file_storage, usuario_id="default", frame_name=None):
    
    frame_bytes = file_storage.read()
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "No se pudo leer la imagen"}

    # Simular detección (REEMPLAZA por detección real con tu modelo YOLOv5)
    predicciones = procesar_imagen(frame) # type: ignore

     # ⚠️ Usar el frame_name recibido, no sobrescribirlo
    if not frame_name:
        # Opcional: generar nombre por hash si no se recibe
        import hashlib
        frame_hash = hashlib.md5(frame_bytes).hexdigest()
        frame_name = frame_hash + ".jpg"

    # Procesar y actualizar métricas
    cliente_monitoreo.procesar_resultados(usuario_id, predicciones, frame_name=frame_name)

    # Obtener resultados
    return cliente_monitoreo.obtener_métricas_finales(usuario_id)



# Función mock de predicción
def simular_prediccion(frame):
    clases = ["bueno", "fisura", "rotura"]
    import random
    return [random.choice(clases) for _ in range(random.randint(1, 3))]

# Función para reiniciar monitoreo
def reset_monitoreo(usuario_id="default"):
    cliente_monitoreo.resetear(usuario_id)
