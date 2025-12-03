import os
import time
import json
import logging
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage, bigquery, tasks_v2
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

# Config global
GCP_PROJECT = "extrac-datos-geosys-production"
GCP_REGION = "us-central1"
CLOUD_TASK_QUEUE = "online-sismica"
WORKER_URL = "https://online-sismica-386277896892.us-central1.run.app/process_single" #pega la ruta de cloud run luego de hacer deploy junto con /process_single

# Inicializar Vertex AI y logging
init(project=GCP_PROJECT, location="us-central1")
logging.basicConfig(level=logging.INFO)

# Listas personalizadas
contratista = pd.read_csv("contratista.csv").iloc[:, 0].dropna().unique().tolist()
operadora = pd.read_csv("operadora.csv").iloc[:, 0].dropna().unique().tolist()
tipo_procesamiento = pd.read_csv("tipo_procesamiento.csv").iloc[:, 0].dropna().unique().tolist()

# Prompt base (omitido por brevedad)
prompt_base = """
Eres un sistema experto en extracción de información estructurada desde documentos técnicos sísmicos. Tu objetivo es extraer con precisión los campos especificados y devolver un objeto JSON con los valores extraídos. Sigue todas las instrucciones al pie de la letra. Devuelve únicamente JSON válido. No expliques ni añadas comentarios.

                            INSTRUCCIONES GENERALES:

                            1. Analiza completamente el documento escaneado o imagen antes de comenzar.
                            2. Detecta zonas clave con información técnica: recuadros, encabezados, pies de página, tablas, márgenes o secciones etiquetadas.
                            3. Aplica correcciones comunes de OCR:
                            - Errores frecuentes: “O” ↔ “0”, “I” ↔ “1”, “B” ↔ “8”, “S” ↔ “5”, “Z” ↔ “2”, “3” ↔ “8”.
                            4. Si hay múltiples candidatos para un campo, prioriza:
                            - Cercanía a la etiqueta
                            - Alineación (a la derecha o en la misma línea de la etiqueta)
                            - Jerarquía visual
                            5. Usa formatos estandarizados:
                            - Fechas en AAAA-MM-DD
                            - Campos nulos como null (sin comillas)
                            6. Devuelve únicamente el objeto JSON especificado más abajo. Sin explicaciones adicionales.
                            7. Si un valor no cumple las reglas, asigna null.

                            CAMPOS A EXTRAER:

                            1. **nombre_de_linea**:  
                            - Etiquetas válidas: “LINE”, “LINE NAME”, “LÍNEA”, “LINE NO.” (no se aceptan abreviaciones)  
                            - Extrae el valor inmediatamente junto a la etiqueta.  
                            - Ignora líneas inferiores o valores no asociados directamente.

                            2. **procesado_por**:  
                            - Usa la lista {contratistas_list}  
                            - Coincidencia por nombre con la lista, ignorando mayúsculas/minúsculas y errores OCR menores  
                            - Si no se encuentra → null

                            3. **procesado_para**:  
                            - Usa la lista {operadoras_list}  
                            - Reglas iguales a "procesado_por"

                            4. **fecha**:  
                            - Etiquetas: "FECHA", "DATE", "PROCESSING DATE", "DISPLAY DATE"  
                            - Formato: AAAA-MM-DD  
                            - Si solo hay mes y año → usa día 01.  
                            - Si el formato es ambiguo (ej. 05/06/97), analiza el contexto para determinar el orden.

                           5. **shot_point_range**:  
                            - Extrae un rango en formato: número-número (por ejemplo, 100-298 o 001-1200).  
                            - Solo considera valores que estén explícitamente etiquetados como:
                                - “SP”
                                - “SHOT POINT”
                                - “STA”
                            - No extraigas rangos que estén junto a otras etiquetas diferentes (como "STATION", "CHANNEL", "LINE", etc.).
                            - Si hay múltiples coincidencias:
                                - Prioriza aquellas con etiquetas en mayúsculas exactas (ej. “SP” sobre “Sp” o “Shot Point”)
                            - Rechaza cualquier valor que:
                                - Esté fuera del formato esperado
                                - Tenga letras, símbolos o espacios entre los números
                            - Si no se encuentra un rango válido y etiquetado → asigna `null`

                            6. **tipo_procesamiento**:  
                            - Usa la lista {tipos_procesamiento_list}  
                            - Si no se encuentra coincidencia → null

                            7. **codigo**:  
                            - Debe comenzar con "SG" o "SGP" seguido de 4 a 6 dígitos.  
                            - Rechaza códigos con guiones, espacios o letras en la parte numérica.  
                            - Ubicación típica: cerca del logo, código de barras, encabezado o pie de página.  
                            - Ejemplos válidos: SG00577, SGP21081  
                            - Ejemplos inválidos: RIB93-26 → null

			                8. **registrado_por**:
                            - Busca en los campos con nombre como: "FIELD PARAMETERS", "ACQUISITION PARAMETERS", "CAMPOS DE ADQUISICIÓN" o "PARAMETROS DE ADQUISICIÓN".
                            - Busca los datos que están presididos por las palabras clave: "RECORDED BY", "SHOT BY" , "ACQUIRED BY" o simplemente "ACQUIRED"
                            - Formato: string, nombre propio de persona, compañia o empresa.
                            - Si no se encuentra → null
                            - no hagas diferencia entre letras mayúsculas o minúsculas.

                            9. **fecha_registro**:
                            - Busca en los campos con nombre como: "FIELD PARAMETERS", "ACQUISITION PARAMETERS", "CAMPOS DE ADQUISICIÓN" o "PARAMETROS DE ADQUISICIÓN".
                            - Busca los datos que están presididos por las palabras clave: "RECORDING DETAILS", "RECORDING TECHNIQUE", "RECORDING DATE", "ACQUIRED" o simplemente "DATE".
                            - Formato AAAA-MM-DD.
                            - Si no tiene fecha de día, extrae la fecha tal y como lo encontraste (ej. 24-junio-2025 -> 2025-06-24; May. 25-> may 25)
                                - Si el formato es ambiguo (ej. 05/06/97), analiza el contexto para determinar el orden.
                            - No hagas diferencia entre minúsculas y mayúsculas.

                            FORMATO DE SALIDA:

                            Devuelve el siguiente formato (exactamente con esta estructura). No añadas comentarios ni texto adicional.

                            {{
                            "nombre_de_linea": null,
                            "procesado_por": null,
                            "procesado_para": null,
                            "fecha": null,
                            "shot_point_range": null,
                            "tipo_procesamiento": null,
                            "codigo": null,
                            "registrado_por": null,
                            "fecha_registro": null
                            }}
"""  # Usa el prompt completo que ya tienes

def build_prompt():
    return prompt_base.format(
        tipos_procesamiento_list=tipo_procesamiento,
        operadoras_list=operadora,
        contratistas_list=contratista
    )

def download_blob_as_bytes(bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes(), blob.content_type

def generate_from_document(document1, prompt, model_version):
    model = GenerativeModel(model_version)
    responses = model.generate_content(
        [document1, prompt],
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 0,
            "top_p": 0.95,
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
    )
    return responses.text.strip()

def save_to_bigquery(file_name, respuesta_texto):    
    client = bigquery.Client()
    table_id = f"{client.project}.gf_pozos.resultados_pozos"

    try:
        client.get_table(table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("archivo", "STRING"),
            bigquery.SchemaField("respuesta_modelo", "STRING"),
            bigquery.SchemaField("fecha_procesamiento", "TIMESTAMP"),  # NUEVO
        ]
        client.create_table(bigquery.Table(table_id, schema=schema))

    query = f"SELECT COUNT(*) as total FROM {table_id} WHERE archivo = @archivo"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("archivo", "STRING", file_name)]
    )
    existe = next(client.query(query, job_config=job_config).result()).total > 0

    if existe:
        update_query = f"""
            UPDATE {table_id} 
            SET respuesta_modelo = @respuesta, 
                fecha_procesamiento = @fecha 
            WHERE archivo = @archivo
        """
        update_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("archivo", "STRING", file_name),
                bigquery.ScalarQueryParameter("respuesta", "STRING", respuesta_texto),
                bigquery.ScalarQueryParameter("fecha", "TIMESTAMP", datetime.utcnow()),
            ]
        )
        client.query(update_query, job_config=update_config).result()
    else:
        client.insert_rows_json(table_id, [{
            "archivo": file_name, 
            "respuesta_modelo": respuesta_texto,
            "fecha_procesamiento": datetime.utcnow().isoformat()
        }])


def save_metrics_to_bigquery(file_name, status, error_msg=None, tiempo_procesamiento=None, model_version=None):
    """
    Guarda métricas de procesamiento en BigQuery
    """
    client = bigquery.Client()
    table_id = f"{client.project}.gf_pozos.metricas_procesamiento"
    
    try:
        client.get_table(table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("archivo", "STRING"),
            bigquery.SchemaField("fecha_procesamiento", "TIMESTAMP"),
            bigquery.SchemaField("status", "STRING"),  # 'success' o 'error'
            bigquery.SchemaField("error_mensaje", "STRING"),
            bigquery.SchemaField("tiempo_procesamiento_seg", "FLOAT"),
            bigquery.SchemaField("model_version", "STRING"),
        ]
        client.create_table(bigquery.Table(table_id, schema=schema))
    
    row = {
        "archivo": file_name,
        "fecha_procesamiento": datetime.utcnow().isoformat(),
        "status": status,
        "error_mensaje": error_msg,
        "tiempo_procesamiento_seg": tiempo_procesamiento,
        "model_version": model_version
    }
    
    client.insert_rows_json(table_id, [row])

# Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "API activa", 200

@app.route("/enqueue_tasks", methods=["POST"])
def enqueue_tasks():
    try:
        data = request.get_json()
        bucket_path = data["bucket_path"]
        cantidad = int(data.get("cantidad", 10))
        model_version = data.get("model_version", "gemini-2.5-flash")

        bucket_name, prefix = bucket_path.replace("gs://", "").split("/", 1)
        client = storage.Client()
        blobs = list(client.list_blobs(bucket_name, prefix=prefix))
        blobs = [b.name for b in blobs if b.name.endswith((".pdf", ".jpg", ".png", ".tiff", ".tif"))][:cantidad]

        task_client = tasks_v2.CloudTasksClient()
        parent = task_client.queue_path(GCP_PROJECT, GCP_REGION, CLOUD_TASK_QUEUE)

        for blob_name in blobs:
            payload = {
                "bucket_name": bucket_name,
                "blob_name": blob_name,
                "model_version": model_version
            }
            task = {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": WORKER_URL,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(payload).encode()
                }
            }
            task_client.create_task(parent=parent, task=task)

        return jsonify({"tareas_enviadas": len(blobs)}), 200
    except Exception as e:
        logging.exception("[ERROR] enqueue_tasks")
        return jsonify({"error": str(e)}), 500

@app.route("/process_single", methods=["POST"])
def process_single():
    start_time = time.time()
    data = request.get_json()
    bucket_name = data["bucket_name"]
    blob_name = data["blob_name"]
    model_version = data.get("model_version", "gemini-2.5-flash")
    
    try:
        # Procesamiento
        file_bytes, mime_type = download_blob_as_bytes(bucket_name, blob_name)
        part = Part.from_data(mime_type=mime_type, data=file_bytes)
        prompt = build_prompt()
        respuesta = generate_from_document(part, prompt, model_version)
        
        # Guardar resultado
        save_to_bigquery(blob_name, respuesta)
        
        # Guardar métricas de éxito
        tiempo_procesamiento = time.time() - start_time
        save_metrics_to_bigquery(
            file_name=blob_name,
            status="success",
            tiempo_procesamiento=tiempo_procesamiento,
            model_version=model_version
        )
        
        logging.info(f"✓ Procesado exitoso: {blob_name} en {tiempo_procesamiento:.2f}s")
        return jsonify({
            "procesado": blob_name,
            "tiempo_seg": tiempo_procesamiento
        }), 200

    except Exception as e:
        # Guardar métricas de error
        tiempo_procesamiento = time.time() - start_time
        error_msg = str(e)
        
        save_metrics_to_bigquery(
            file_name=blob_name,
            status="error",
            error_msg=error_msg,
            tiempo_procesamiento=tiempo_procesamiento,
            model_version=model_version
        )
        
        logging.exception(f"✗ Error procesando {blob_name}")
        return jsonify({
            "error": error_msg,
            "archivo": blob_name
        }), 500
		
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
