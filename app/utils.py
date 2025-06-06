import os
from pathlib import Path
import torch
from PIL import Image
from fpdf import FPDF
import gdown
import zipfile

# ---------------------- Modelo YOLOv5 -----------------------

# Descargar el modelo desde Google Drive si no existe
def descargar_modelo():
    modelo_path = "app/best50e1.pt"
    if not os.path.exists(modelo_path):
        print("🔽 Descargando modelo YOLOv5 desde Google Drive...")
        url = "https://drive.google.com/uc?id=174Td9kRd10iImunxIwrXZsKn9PduBDTX"
        gdown.download(url, modelo_path, quiet=False)
        print("✅ Modelo descargado.")
    return modelo_path

# Cargar modelo YOLOv5 directamente desde archivo
def cargar_modelo():
    modelo_path = descargar_modelo()
    modelo = torch.load(modelo_path, map_location=torch.device('cpu'))  # usa 'cuda' si quieres forzar GPU
    modelo.eval()
    return modelo

# Cargar modelo global
modelo = cargar_modelo()

# ---------------------- Procesamiento de imagen -----------------------

def procesar_imagen(frame_array):
    img = Image.fromarray(frame_array).convert('RGB')
    results = modelo(img)
    predicciones = results.pandas().xyxy[0]['name'].tolist()
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
