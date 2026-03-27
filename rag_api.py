from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import os
import requests as http_requests
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configurar pgvector como base de datos vectorial
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
TABLE_NAME = os.getenv("TABLE_NAME", "rag_embeddings")
EMBED_DIM = int(os.getenv("EMBED_DIM", 768))
OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Configurar el LLM y los embeddings
Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0, additional_kwargs={"num_predict": 300})
Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
Settings.chunk_size = 512
Settings.chunk_overlap = 64

vector_store = PGVectorStore.from_params(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    table_name=TABLE_NAME,
    embed_dim=EMBED_DIM,
)

# Comprobar si ya hay embeddings guardados en la BD
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
cur = conn.cursor()
cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", (f"data_{TABLE_NAME}",))
table_exists = cur.fetchone()[0]
row_count = 0
if table_exists:
    cur.execute(f"SELECT COUNT(*) FROM data_{TABLE_NAME}")
    row_count = cur.fetchone()[0]
cur.close()
conn.close()

if table_exists and row_count > 0:
    print(f"Cargando índice existente desde pgvector ({row_count} fragmentos)...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
else:
    docs_path = os.getenv("DOCS_DIR", "../rag_books/docs")
    print("Cargando documentos y generando embeddings...")
    documents = SimpleDirectoryReader(docs_path).load_data()
    print(f"Documentos cargados: {len(documents)}")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("Embeddings guardados en PostgreSQL.")

# Retriever para buscar chunks relevantes
retriever = index.as_retriever(similarity_top_k=4)

FRASES_A_ELIMINAR = [
    "Según el contexto, ", "Según el contexto proporcionado, ",
    "Basándome en el contexto, ", "Basándome en la información proporcionada, ",
    "De acuerdo con el contexto, ", "El contexto indica que ",
    "El texto indica que ", "El texto menciona que ",
    "El texto no proporciona información sobre ", "El texto no proporciona ",
    "La información proporcionada no incluye ", "No se proporciona información sobre ",
    "According to the context, ", "Based on the context, ",
    "según el contexto, ", "basándome en el contexto, ",
    "el texto no proporciona información sobre ", "el texto no proporciona ",
]


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "La pregunta no puede estar vacía"}), 400

    # Recuperar chunks relevantes
    nodes = retriever.retrieve(question)
    context = "\n\n".join([n.text for n in nodes])

    # Prompt directo a Ollama
    prompt = (
        f"Lee este texto:\n"
        f"---\n{context}\n---\n\n"
        f"Usando SOLO el texto de arriba, responde en español: {question}\n"
        f"Respuesta:"
    )

    def generate():
        resp = http_requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": True, "options": {"num_predict": 300}},
            stream=True,
            timeout=180
        )
        buffer = ""
        for line in resp.iter_lines():
            if line:
                obj = json.loads(line)
                token = obj.get("response", "")
                if token:
                    buffer += token
                    for frase in FRASES_A_ELIMINAR:
                        buffer = buffer.replace(frase, "")
                    yield f"data: {json.dumps({'token': token})}\n\n"
                if obj.get("done"):
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    break

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
