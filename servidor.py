import os
import json
import time
import sqlite3
import httpx
import re
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

# --- Configuración y Carpetas ---
app = FastAPI()
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
DIR_SESIONES = "sesiones_guardadas"
DIR_PROMPTS = "prompts"
DB_FILE = "experimentos.db"

# Creación automática de directorios
os.makedirs(DIR_SESIONES, exist_ok=True)
os.makedirs(DIR_PROMPTS, exist_ok=True)

# Crear archivo de prueba para que tengas algo en la lista de prompts
if not os.listdir(DIR_PROMPTS):
    with open(os.path.join(DIR_PROMPTS, "experto_ciber.txt"), "w", encoding="utf-8") as f:
        f.write("Eres un experto senior en ciberseguridad. Respondes con precisión técnica, sin saludos y directo al punto.")

# --- Base de Datos (SQLite) ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sesiones
                 (id_sesion TEXT PRIMARY KEY, titulo_ia TEXT, subtitulo_humano TEXT, reacciones TEXT, fecha TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS interacciones
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, id_sesion TEXT, turno INTEGER, 
                  modelo TEXT, prompt_usuario TEXT, respuesta_modelo TEXT, tiempo_s REAL, tps REAL, 
                  FOREIGN KEY(id_sesion) REFERENCES sesiones(id_sesion))''')
    conn.commit()
    conn.close()

init_db()

# --- Modelos Pydantic ---
class InferenciaPayload(BaseModel):
    modelo: str
    system_prompt: str
    origen_prompt: str
    mensajes_historial: List[Dict[str, str]]
    prompt_actual: str
    temperatura: float
    top_p: float
    top_k: int
    max_tokens: int

class FinalizarPayload(BaseModel):
    id_sesion: str
    subtitulo_humano: str
    reacciones_viscerales: str
    historial_completo: List[Dict[str, Any]]
    parametros_finales: Dict[str, Any]
    metadatos_modelo: Dict[str, Any]

# --- Utilidades ---
def extraer_metadatos_gguf(model_id: str) -> Dict[str, str]:
    """Extrae cuantización y tamaño estimado basado en el nombre del modelo."""
    meta = {"cuantizacion": "Desconocida", "tamano": "Desconocido", "arquitectura": "Auto"}
    
    # Buscar cuantización (ej: Q4_K_M, Q8_0)
    q_match = re.search(r'(Q\d_[K_A-Z0-9]+)', model_id, re.IGNORECASE)
    if q_match:
        meta["cuantizacion"] = q_match.group(1).upper()
        
    # Buscar tamaño en billones de parámetros (ej: 8B, 72b)
    b_match = re.search(r'(\d+(?:\.\d+)?[Bb])', model_id)
    if b_match:
        meta["tamano"] = b_match.group(1).upper()
        
    # Inferir contexto máximo basado en familia de modelos comunes
    if "llama-3" in model_id.lower():
        meta["contexto_max"] = 8192
    elif "qwen2" in model_id.lower() or "qwen-2" in model_id.lower():
        meta["contexto_max"] = 32768
    else:
        meta["contexto_max"] = 4096 # Fallback general
        
    return meta

# --- Rutas de la API ---
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/modelos")
async def get_modelos():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{LM_STUDIO_URL}/models")
            data = resp.json()
            modelos_enriquecidos = []
            for m in data.get("data", []):
                m_data = {"id": m["id"], "metadatos": extraer_metadatos_gguf(m["id"])}
                modelos_enriquecidos.append(m_data)
            return {"data": modelos_enriquecidos}
        except Exception as e:
            return {"data": [{"id": "Error de conexión", "metadatos": {}}]}

@app.get("/api/prompts")
async def listar_prompts():
    archivos = [f for f in sorted(os.listdir(DIR_PROMPTS)) if f.endswith(".txt")]
    return {"prompts": archivos}

@app.get("/api/prompts/{nombre}")
async def cargar_prompt(nombre: str):
    ruta = os.path.join(DIR_PROMPTS, nombre)
    if not os.path.isfile(ruta):
        return {"error": "Archivo no encontrado"}
    with open(ruta, "r", encoding="utf-8") as f:
        return {"contenido": f.read()}

@app.post("/api/inferencia")
async def inferencia(payload: InferenciaPayload, request: Request):
    mensajes = [{"role": "system", "content": payload.system_prompt}]
    mensajes.extend(payload.mensajes_historial)
    mensajes.append({"role": "user", "content": payload.prompt_actual})

    lm_payload = {
        "model": payload.modelo,
        "messages": mensajes,
        "temperature": payload.temperatura,
        "top_p": payload.top_p,
        "top_k": payload.top_k,
        "max_tokens": payload.max_tokens,
        "stream": True
    }

    async def stream_generator():
        start_time = time.time()
        tokens_generados = 0
        texto_completo = ""
        
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", f"{LM_STUDIO_URL}/chat/completions", json=lm_payload) as response:
                    async for chunk in response.aiter_lines():
                        if await request.is_disconnected():
                            break # AbortController disparado desde el frontend
                        
                        if chunk.startswith("data: ") and chunk != "data: [DONE]":
                            try:
                                data = json.loads(chunk[6:])
                                token = data['choices'][0]['delta'].get('content', '')
                                if token:
                                    tokens_generados += 1
                                    texto_completo += token
                                    yield token
                            except:
                                pass
            except Exception:
                pass
        
        tiempo_total = time.time() - start_time
        tps = tokens_generados / tiempo_total if tiempo_total > 0 else 0
        yield f"\n[METRICAS_FINAL]||{round(tiempo_total, 2)}||{round(tps, 2)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.post("/api/sesiones/finalizar")
async def finalizar_sesion(payload: FinalizarPayload):
    # Generar título automático
    prompt_resumen = "Resume toda nuestra conversación anterior en un título técnico de máximo 5 palabras. Responde SOLO con el título, nada más."
    mensajes = [{"role": "system", "content": "Eres un titulador de documentos."}]
    mensajes.extend([{"role": m["role"], "content": m["content"]} for m in payload.historial_completo if "role" in m])
    mensajes.append({"role": "user", "content": prompt_resumen})
    
    titulo_ia = "Sesión Experimental"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{LM_STUDIO_URL}/chat/completions", json={
                "model": payload.parametros_finales.get("modelo", ""),
                "messages": mensajes,
                "temperature": 0.3,
                "max_tokens": 20
            })
            titulo_ia = resp.json()['choices'][0]['message']['content'].strip(' "')
    except Exception:
        pass

    # JSON Completo
    sesion_data = {
        "id_sesion": payload.id_sesion,
        "fecha": datetime.now().isoformat(),
        "titulo_ia": titulo_ia,
        "subtitulo_humano": payload.subtitulo_humano,
        "reacciones_viscerales": payload.reacciones_viscerales,
        "modelo_activo": {
            "id": payload.parametros_finales.get("modelo"),
            "metadata": payload.metadatos_modelo
        },
        "interacciones": payload.historial_completo
    }

    # Dual-Write: Guardar JSON
    ruta_json = os.path.join(DIR_SESIONES, f"{payload.id_sesion}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(sesion_data, f, indent=4, ensure_ascii=False)

    # Dual-Write: SQLite
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO sesiones VALUES (?, ?, ?, ?, ?)", 
              (payload.id_sesion, titulo_ia, payload.subtitulo_humano, payload.reacciones_viscerales, sesion_data["fecha"]))
    
    for turno_idx, inter in enumerate(payload.historial_completo):
        if inter.get("role") == "user":
            respuesta = ""
            for next_inter in payload.historial_completo[turno_idx+1:]:
                if next_inter.get("role") == "assistant":
                    respuesta = next_inter.get("content", "")
                    break
            c.execute("INSERT INTO interacciones (id_sesion, turno, modelo, prompt_usuario, respuesta_modelo, tiempo_s, tps) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (payload.id_sesion, turno_idx, payload.parametros_finales.get("modelo"), inter.get("content"), respuesta, 
                       inter.get("metricas", {}).get("tiempo_s", 0), inter.get("metricas", {}).get("tps", 0)))
    conn.commit()
    conn.close()

    return {"status": "ok", "titulo_ia": titulo_ia}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
