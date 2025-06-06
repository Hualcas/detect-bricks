import os
from pathlib import Path
import torch
from PIL import Image
from fpdf import FPDF
import gdown
import zipfile
from ultralytics import YOLO  # ✅ Usamos YOLOv8

# ---------------------- Modelo YOLOv5 -----------------------

modelo = None  # 🔁 Se cargará bajo demanda

# Descargar el modelo desde Google Drive si no existe
def descargar_modelo():
    modelo_path = "app/best50e1.pt"
    if not os.path.exists(modelo_path):
        print("🔽 Descargando modelo YOLOv5 desde Google Drive...")
        url = "https://drive.google.com/uc?id=1YkkXzRnnUn_AO7J7kw78lPppEb6-2g7G"
        gdown.download(url, modelo_path, quiet=False)
        print("✅ Modelo descargado.")
    return modelo_path

# Cargar modelo YOLOv5 directamente desde archivo
def cargar_modelo():
    modelo_path = descargar_modelo()
    modelo = YOLO(modelo_path)  # ✅ Carga directa con ultralytics
    return modelo

# ---------------------- Procesamiento de imagen -----------------------

def procesar_imagen(frame_array):
    global modelo
    if modelo is None:
        modelo = cargar_modelo()  # ✅ Solo se carga una vez cuando se necesita

    img = Image.fromarray(frame_array).convert('RGB')
    results = modelo.predict(img, verbose=False)  # ✅ Usamos .predict()
    predicciones = []
    if results and results[0].names:
        names_dict = results[0].names
        for r in results[0].boxes.cls.tolist():
            predicciones.append(names_dict[int(r)])
    return predicciones

# ---------------------- Descarga de imágenes -----------------------

def descargar_imagenes():
    if not os.path.exists("imagenes"):
        print("🔽 Descargando imágenes desde Google Drive...")
        zip_path = "imagenes.zip"
        url = "https://drive.google.com/uc?id=1vTG9F8YChZtPUCWNtJX7rk2_zC8Y1JyG"
        gdown.download(url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("imagenes")
        print("✅ Imágenes extraídas.")

# ---------------------- Generación de reportes PDF -----------------------

def generar_pdf(monitoreo, id_reporte=None):
    os.makedirs("app/static/reportes", exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=" Reporte de Monitoreo", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f" Fecha: {monitoreo.fecha}", ln=True)
    pdf.cell(200, 10, txt=f" Inicio: {monitoreo.hora_inicio}", ln=True)
    pdf.cell(200, 10, txt=f" Fin: {monitoreo.hora_fin}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f" Total Ladrillos: {monitoreo.total_ladrillos}", ln=True)
    pdf.cell(200, 10, txt=f" Buenos: {monitoreo.ladrillos_buenos}", ln=True)
    pdf.cell(200, 10, txt=f" Malos: {monitoreo.ladrillos_malos}", ln=True)
    pdf.cell(200, 10, txt=f" Precisión: {monitoreo.precision:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f" Tiempo Promedio Fisura: {monitoreo.tiempo_promedio_fisura}", ln=True)

    filename = f"reporte_{id_reporte or 'temporal'}.pdf"
    path = os.path.join("app", "static", "reportes", filename)
    pdf.output(path)

    return f"reportes/{filename}"

# ---------------------- Métricas dummy -----------------------

def get_monitoring_results():
    return {
        "total": 100,
        "buenos": 85,
        "malos": 15,
        "precision": 85.0,
        "promedio_tiempo": "0.2s"
    }
