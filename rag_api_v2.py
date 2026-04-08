"""
RAG API v2 - Arquitectura basada en poc-ai:
- psycopg2 directo (sin llama-index)
- pgvector con tabla 'book_chunks' propia
- Ollama API REST directa (sin SDK)
- Chunking propio por párrafos/frases
- Streaming SSE en /chat
"""

import json
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
from translation import translate_ca_to_en, translate_en_to_ca
from psycopg2.extras import execute_values
import numpy as np
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuración ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBED_DIM = int(os.getenv("EMBED_DIM", 768))

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "rag_books", "docs")

CHUNK_SIZE = 500      # caracteres (~100 palabras)
CHUNK_OVERLAP = 100   # caracteres de solapamiento

FRASES_A_ELIMINAR = [
    "According to the context, ", "Based on the context, ",
    "Según el contexto, ", "Según el contexto proporcionado, ",
    "Basándome en el contexto, ", "Basándome en la información proporcionada, ",
    "De acuerdo con el contexto, ", "El contexto indica que ",
    "El texto no proporciona información sobre ", "El texto no proporciona ",
    "según el contexto, ", "basándome en el contexto, ",
]


# --- Base de datos ---
def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS book_chunks (
            id SERIAL PRIMARY KEY,
            document_id TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({EMBED_DIM})
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS book_chunks_embedding_idx ON book_chunks USING ivfflat (embedding vector_cosine_ops);")
    conn.commit()
    cur.close()
    conn.close()


def document_exists(doc_id: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM book_chunks WHERE document_id = %s LIMIT 1", (doc_id,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists


def chunk_count() -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM book_chunks")
    n = cur.fetchone()[0]
    cur.close()
    conn.close()
    return n


# --- Embeddings vía Ollama REST ---
def get_embedding(text: str):
    res = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=30,
    )
    if res.status_code != 200:
        print(f"Error embedding: {res.text}")
        return None
    return res.json().get("embedding")


# --- Chunking semántico (igual que poc-ai) ---
def split_into_sentences(text: str):
    separators = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
    sentences = []
    current = ""
    i = 0
    while i < len(text):
        current += text[i]
        for sep in separators:
            if text[i: i + len(sep)] == sep:
                sentences.append(current.strip())
                current = ""
                i += len(sep) - 1
                break
        i += 1
    if current.strip():
        sentences.append(current.strip())
    return [s for s in sentences if s]


def chunk_text(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > CHUNK_SIZE:
            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > CHUNK_SIZE:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = current_chunk[-CHUNK_OVERLAP:].strip() + " "
                    if len(sentence) > CHUNK_SIZE:
                        words = sentence.split()
                        temp = current_chunk
                        for word in words:
                            if len(temp) + len(word) + 1 > CHUNK_SIZE and temp.strip():
                                chunks.append(temp.strip())
                                temp = temp[-CHUNK_OVERLAP:].strip() + " "
                            temp += word + " "
                        current_chunk = temp
                    else:
                        current_chunk += sentence + " "
                else:
                    current_chunk += sentence + " "
        else:
            if len(current_chunk) + len(paragraph) + 2 > CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = current_chunk[-CHUNK_OVERLAP:].strip() + "\n\n"
                current_chunk += paragraph
            else:
                current_chunk = (current_chunk + "\n\n" + paragraph) if current_chunk else paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


# --- Procesado de documentos ---
def process_documents():
    conn = get_conn()
    cur = conn.cursor()

    files = [f for f in os.listdir(DOCS_DIR) if f.endswith("_en.txt") or f.endswith(".pdf")]
    print(f"Documentos encontrados: {len(files)}")

    for file_name in files:
        file_path = os.path.join(DOCS_DIR, file_name)

        if document_exists(file_name):
            print(f"Ya procesado: {file_name}")
            continue

        print(f"Procesando: {file_name}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            continue

        chunks = chunk_text(text)
        print(f"  - {len(chunks)} chunks generados")

        rows = []
        for chunk in chunks:
            emb = get_embedding(chunk)
            if emb is not None:
                rows.append((file_name, chunk, emb))

        execute_values(cur, "INSERT INTO book_chunks (document_id, text, embedding) VALUES %s", rows)
        print(f"  - Guardado en DB: {file_name}")

    conn.commit()
    cur.close()
    conn.close()
    print("Proceso completado")


# --- Búsqueda de chunks similares ---
def get_similar_chunks(query: str, top_k: int = 4):
    embedding = get_embedding(query)
    if embedding is None:
        raise ValueError("No se pudo generar el embedding de la consulta")

    vector = np.array(embedding).astype("float32").tolist()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT document_id, text, (embedding <-> %s::vector) AS distance
        FROM book_chunks
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (vector, vector, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"document_id": r[0], "text": r[1], "distance": float(r[2])} for r in rows]


# --- LLM vía Ollama REST (streaming) ---
def generate_answer_stream(query: str, chunks: list):
    if not chunks:
        def no_context():
            msg = "Aquesta informació no figura en els textos que tinc disponibles."
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return no_context()

    context = "\n\n".join(
        [f"[{c['document_id']} - distancia {c['distance']:.3f}]\n{c['text']}" for c in chunks]
    )

    prompt = (
        f"Read this information carefully:\n"
        f"---\n{context}\n---\n\n"
        f"Based ONLY on the information above, answer in English as a literary expert.\n"
        f"Be direct and specific. Mention names, dates and details from the text.\n"
        f"Do not say 'the context' or 'according to'. Answer directly.\n"
        f"If the answer is not in the text above, say: "
        f"'That information is not found in the texts available to me'.\n\n"
        f"Question: {query}\n"
        f"Answer in English:"
    )

    def stream_generator():
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 400, "temperature": 0.3, "top_p": 0.9},
                },
                timeout=180,
            )
            answer_en = resp.json()["response"].strip()
            answer_ca = translate_en_to_ca(answer_en)
            words = answer_ca.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'token': f'Error: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

    return stream_generator()


# --- Endpoints Flask ---

@app.route("/health", methods=["GET"])
def health():
    n = chunk_count()
    return jsonify({"status": "ok", "chunks_in_db": n, "model": LLM_MODEL})


@app.route("/process-docs", methods=["POST"])
def process_docs_api():
    """Procesa y guarda en pgvector todos los docs de rag_books/docs/ que no estén ya."""
    try:
        process_documents()
        return jsonify({"message": "Documentos procesados", "chunks": chunk_count()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint RAG con streaming SSE. Body: {"question": "..."}"""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "La pregunta no puede estar vacía"}), 400

    try:
        question_en = translate_ca_to_en(question)
        chunks = get_similar_chunks(question_en, top_k=4)
        return Response(
            stream_with_context(generate_answer_stream(question_en, chunks)),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/similar-chunks", methods=["POST"])
def similar_chunks_api():
    """Debug: devuelve los chunks más similares a una query. Body: {"query": "...", "top_k": 4}"""
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 4)
    if not query:
        return jsonify({"error": "Falta 'query'"}), 400
    try:
        chunks = get_similar_chunks(query, top_k)
        return jsonify({"chunks": chunks}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Arranque ---
if __name__ == "__main__":
    print("Inicializando base de datos...")
    init_db()

    n = chunk_count()
    if n == 0:
        print("No hay chunks en la BD, procesando documentos...")
        process_documents()
    else:
        print(f"Cargando índice existente desde pgvector ({n} chunks)...")

    print(f"Modelo LLM: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}")
    app.run(host="0.0.0.0", port=5001, debug=False)
