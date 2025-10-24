import os
import time
import json
import logging
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud import storage, bigquery, tasks_v2
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

# Config global
GCP_PROJECT = "geosys-analitica-dev"
GCP_REGION = "us-east1"
CLOUD_TASK_QUEUE = "online-sismos"
WORKER_URL = "https://online-sismos-439188544158.us-east1.run.app/process_single"

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

                            7. **lote**:  
                            - Etiquetas: “BLOCK”, “LOTE”, “BLOCKS”  
                            - Extrae directamente de la etiqueta  
                            - Si es ambiguo o no aparece → null

                            8. **analisis_velocidades**:  
                            - true si:
                                - Hay múltiples tablas pequeñas con decimales
                                - O aparece la etiqueta “VELOCITY ANALYSIS”  
                            - Si no → false

                            9. **codigo**:  
                            - Debe comenzar con "SG" o "SGP" seguido de 4 a 6 dígitos.  
                            - Rechaza códigos con guiones, espacios o letras en la parte numérica.  
                            - Ubicación típica: cerca del logo, código de barras, encabezado o pie de página.  
                            - Ejemplos válidos: SG00577, SGP21081  
                            - Ejemplos inválidos: RIB93-26 → null

                            10. **intervalo_de_receptores**:  
                            - Palabras clave: “station interval”, “receiver interval”, “group interval”  
                            - Formato: número + unidad  
                            - Si no se encuentra → null

                            11. **intervalos_de_fuentes**:  
                            - Palabras clave: “shot interval”, “SP interval”, “Source interval”  
                            - Formato: número + unidad (ej. 25 m)  
                            - Si no se encuentra → null

                            12. **datum**:  
                            - Etiquetas: “DATUM”, “DATUM LEVEL”, “DATUM ELEVATION”  
                            - Suele estar alineado a la derecha de la etiqueta.  
                            - Si es poco confiable → null

                            13. **station**:  
                            - Formato preferido: “200-51”  
                            - Estrategia de extracción:
                                1. Si existe la etiqueta “STATION”, localiza la lista numérica asociada directamente (tabla, fila o columna próxima) y extrae el primer y último número como rango.
                                2. Si no se encuentra “STATION”:
                                    - Identifica los valores extremos (primer y último) de las coordenadas en la gráfica de línea sísmica o tablas relacionadas con el trazado horizontal.
                                    - Extrae esos dos valores como rango numérico.
                            - Si no puede determinarse por ninguna de las vías → null

                            14. **velocidad_de_reemplazamiento**:  
                            - Palabras clave: “Replacement velocity”, “Correctional velocity”  
                            - Formato: número + unidad (ej. 2000 m/s)  
                            - Si no se encuentra → null

                            15. **dominio_profundidad**:  
                            - Busca en “PROCESSING SEQUENCE” o “Processing sequence”  
                            - Si contiene la palabra “depth” → true, si no → false

                            FORMATO DE SALIDA:

                            Devuelve el siguiente formato (exactamente con esta estructura). No añadas comentarios ni texto adicional.

                            {{
                            "nombre_de_linea": null,
                            "procesado_por": null,
                            "procesado_para": null,
                            "fecha": null,
                            "shot_point_range": null,
                            "tipo_procesamiento": null,
                            "lote": null,
                            "analisis_velocidades": null,
                            "codigo": null,
                            "intervalo_de_receptores": null,
                            "intervalos_de_fuentes": null,
                            "datum": null,
                            "station": null,
                            "velocidad_de_reemplazamiento": null,
                            "dominio_profundidad": null
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
    table_id = f"{client.project}.online_sismos.resultados"

    try:
        client.get_table(table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("archivo", "STRING"),
            bigquery.SchemaField("respuesta_modelo", "STRING"),
        ]
        client.create_table(bigquery.Table(table_id, schema=schema))

    query = f"SELECT COUNT(*) as total FROM {table_id} WHERE archivo = @archivo"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("archivo", "STRING", file_name)]
    )
    existe = next(client.query(query, job_config=job_config).result()).total > 0

    if existe:
        update_query = f"UPDATE {table_id} SET respuesta_modelo = @respuesta WHERE archivo = @archivo"
        update_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("archivo", "STRING", file_name),
                bigquery.ScalarQueryParameter("respuesta", "STRING", respuesta_texto),
            ]
        )
        client.query(update_query, job_config=update_config).result()
    else:
        client.insert_rows_json(table_id, [{"archivo": file_name, "respuesta_modelo": respuesta_texto}])

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
        blobs = [b.name for b in blobs if b.name.endswith((".pdf", ".jpg", ".png", ".tiff"))][:cantidad]

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
    try:
        data = request.get_json()
        bucket_name = data["bucket_name"]
        blob_name = data["blob_name"]
        model_version = data.get("model_version", "gemini-2.5-flash")

        file_bytes, mime_type = download_blob_as_bytes(bucket_name, blob_name)
        part = Part.from_data(mime_type=mime_type, data=file_bytes)
        prompt = build_prompt()
        respuesta = generate_from_document(part, prompt, model_version)
        save_to_bigquery(blob_name, respuesta)
        return jsonify({"procesado": blob_name}), 200

    except Exception as e:
        logging.exception("[ERROR] process_single")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
