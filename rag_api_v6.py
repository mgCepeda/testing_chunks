"""
RAG API v6 - Pipeline Ca→En→Ca (cross-lingual RAG):
  1. El usuario pregunta en catalán
  2. La pregunta se traduce al inglés antes de hacer la búsqueda semántica
  3. Los documentos están indexados en inglés (ficheros *_en.txt)
  4. El LLM genera la respuesta en inglés (máxima calidad)
  5. La respuesta inglesa se traduce al catalán antes de devolverla al usuario

Ventaja vs versiones anteriores:
  Los LLMs pequeños (mistral, qwen2, phi3...) funcionan mucho mejor en inglés:
  - Menos alucinaciones
  - Respuestas más precisas y completas
  - Sin mezcla de idiomas en la respuesta

Estrategia de chunking: SemanticSplitterNodeParser (igual que v5)
Puerto: 5006
Tabla pgvector: llamaindex_semantic_chunks_en
"""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS

from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import MetadataMode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

app = Flask(__name__)
CORS(app)

# --- Configuración ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBED_DIM = int(os.getenv("EMBED_DIM", 768))

# Directorio con los documentos en INGLÉS (*_en.txt)
DOCS_DIR = os.path.join(os.path.dirname(__file__), "rag_books", "docs")
DOCS_PATTERN = "*_en.txt"

BREAKPOINT_PERCENTILE = 70

# --- Estado global ---
_index = None


def get_vector_store():
    return PGVectorStore.from_params(
        **DB_CONFIG,
        table_name="llamaindex_semantic_chunks_en",
        embed_dim=EMBED_DIM,
    )


def count_rows():
    import psycopg2
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM data_llamaindex_semantic_chunks_en")
        n = cur.fetchone()[0]
        cur.close()
        conn.close()
        return n
    except Exception:
        return 0


def build_index():
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    n_rows = count_rows()
    if n_rows > 0:
        print(f"Cargando índice English existente ({n_rows} nodos en pgvector)...")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
    else:
        print("Indexando documentos en inglés por primera vez...")
        # Carga solo los ficheros *_en.txt
        from llama_index.core import SimpleDirectoryReader
        documents = SimpleDirectoryReader(
            DOCS_DIR,
            required_exts=[".txt"],
            filename_as_id=True,
        ).load_data()
        # Filtrar solo los _en.txt
        documents = [d for d in documents if "_en" in d.metadata.get("file_name", "")]
        print(f"  - Documentos en inglés cargados: {len(documents)}")

        node_parser = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=BREAKPOINT_PERCENTILE,
        )
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        print(f"  - Nodos semánticos generados: {len(nodes)}")

        sizes = [len(n.get_content()) for n in nodes]
        if sizes:
            print(f"  - Tamaño chunks: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)} chars")

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
        print("  - Índice English guardado en pgvector")

    return index


# --- Traducción con Ollama ---
def translate_to_english(text: str) -> str:
    """Traduce la pregunta del catalán al inglés."""
    prompt = (
        "Translate the following question from Catalan to English. "
        "Return only the translated question, nothing else.\n\n"
        f"{text}"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def translate_to_catalan(text: str) -> str:
    """Traduce la respuesta del inglés al catalán."""
    prompt = (
        "Translate the following text from English to Catalan. "
        "Return only the translated text, with no explanation or commentary. "
        "Preserve proper nouns (character names, place names).\n\n"
        f"{text}"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# --- Generación con Ollama (respuesta en inglés) ---
def generate_answer_english(query_en: str, nodes: list) -> str:
    MIN_SCORE = 0.65
    nodes = [n for n in nodes if n.score is None or n.score >= MIN_SCORE]

    if not nodes:
        return "This information is not available in the documents provided."

    context_parts = []
    for node in nodes:
        source = node.metadata.get("file_name", "unknown")
        text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        context_parts.append(f"[{source}]\n{text}")
    context = "\n\n".join(context_parts)

    prompt = (
        f"Relevant context:\n{context}\n\n"
        f"User question: {query_en}\n\n"
        f"Answer in English with all the details from the context: "
        f"names, dates, places and specific facts. "
        f"If the context does not contain relevant information, say so clearly.\n"
        f"Answer:"
    )

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# --- Endpoint /chat ---
@app.route("/chat", methods=["POST"])
def chat():
    global _index
    data = request.get_json(force=True)
    query_ca = data.get("question", "").strip()
    if not query_ca:
        return {"error": "question is required"}, 400

    if _index is None:
        _index = build_index()

    def generate():
        # Paso 1: traducir pregunta Ca → En
        yield f"data: {json.dumps({'token': '', 'status': 'Traduint pregunta...'})}\n\n"
        query_en = translate_to_english(query_ca)
        print(f"  [v6] Query Ca: {query_ca!r}")
        print(f"  [v6] Query En: {query_en!r}")

        # Paso 2: recuperar chunks relevantes (en inglés)
        retriever = _index.as_retriever(similarity_top_k=6)
        nodes = retriever.retrieve(query_en)
        print(f"  [v6] Nodes retrieved: {len(nodes)}")
        for n in nodes:
            print(f"       score={n.score:.3f}  file={n.metadata.get('file_name','?')}")

        # Paso 3: generar respuesta en inglés
        yield f"data: {json.dumps({'token': '', 'status': 'Generant resposta...'})}\n\n"
        answer_en = generate_answer_english(query_en, nodes)
        print(f"  [v6] Answer En: {answer_en[:120]!r}...")

        # Paso 4: traducir respuesta En → Ca
        yield f"data: {json.dumps({'token': '', 'status': 'Traduint resposta...'})}\n\n"
        answer_ca = translate_to_catalan(answer_en)
        print(f"  [v6] Answer Ca: {answer_ca[:120]!r}...")

        # Emitir la respuesta final token a token (simulado por palabras)
        words = answer_ca.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model": LLM_MODEL, "version": "v6", "pipeline": "Ca→En→Ca"}


if __name__ == "__main__":
    print(f"RAG v6 — Pipeline Ca→En→Ca")
    print(f"Modelo LLM: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}")
    print(f"Documentos: {DOCS_DIR} (solo *_en.txt)")
    _index = build_index()
    app.run(host="0.0.0.0", port=5006, debug=False)
