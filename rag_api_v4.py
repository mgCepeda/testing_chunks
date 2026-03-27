"""
RAG API v4 - SentenceWindowNodeParser con llama-index:
- SentenceWindowNodeParser: divide en frases individuales (embedding muy preciso)
  pero guarda una ventana de frases vecinas en los metadatos de cada nodo
- MetadataReplacementPostProcessor: tras recuperar una frase por similitud,
  la sustituye por su ventana completa antes de pasarla al LLM (más contexto)
- PGVectorStore: almacena embeddings de cada frase en pgvector
- Generación: llamada REST directa a Ollama (sin query_engine de llama-index)
- Puerto: 5004

Ventaja vs v3 (HierarchicalNodeParser):
  - Embeddings más precisos (frase individual vs chunk de 256 chars)
  - Sin necesidad de docstore en disco (la ventana viaja en los metadatos del nodo)
  - Más simple: no hay AutoMergingRetriever ni fichero JSON extra
"""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
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

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "rag_books", "docs")

# Ventana: cuántas frases vecinas (a cada lado) almacenar en metadatos
WINDOW_SIZE = 3

FRASES_A_ELIMINAR = [
    "According to the context, ", "Based on the context, ",
    "Según el contexto, ", "Según el contexto proporcionado, ",
    "Basándome en el contexto, ", "Basándome en la información proporcionada, ",
    "De acuerdo con el contexto, ", "El contexto indica que ",
    "El texto no proporciona información sobre ", "El texto no proporciona ",
    "según el contexto, ", "basándome en el contexto, ",
]

# --- Estado global del índice ---
_index = None
_vector_store = None
_processed = False  # True si ya hay datos en pgvector


def get_vector_store():
    return PGVectorStore.from_params(
        **DB_CONFIG,
        table_name="llamaindex_sw_chunks",  # tabla propia para v4
        embed_dim=EMBED_DIM,
    )


def count_rows():
    """Cuenta filas en la tabla pgvector para saber si ya está procesado."""
    import psycopg2
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM data_llamaindex_sw_chunks")
        n = cur.fetchone()[0]
        cur.close()
        conn.close()
        return n
    except Exception:
        return 0


def build_index():
    """
    Construye o carga el índice con SentenceWindowNodeParser:
    1. SentenceWindowNodeParser(window_size=3): cada nodo = 1 frase,
       metadatos contienen la frase + 3 vecinas a cada lado
    2. Todos los nodos se indexan en pgvector (solo la frase, sin ventana)
    3. MetadataReplacementPostProcessor reemplaza la frase por la ventana al recuperar
    """
    global _processed

    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    n_rows = count_rows()
    if n_rows > 0:
        print(f"Cargando índice existente ({n_rows} nodos en pgvector)...")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        _processed = True
    else:
        print("Procesando documentos por primera vez...")
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        print(f"  - Documentos cargados: {len(documents)}")

        # Cada nodo = 1 frase; ventana de WINDOW_SIZE frases a cada lado
        # guardada en metadata["window"] para MetadataReplacementPostProcessor
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=WINDOW_SIZE,
            window_metadata_key="window",       # clave donde se guarda la ventana
            original_text_metadata_key="original_sentence",  # clave de la frase original
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"  - Nodos totales (frases): {len(nodes)}")

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
        _processed = True
        print("  - Índice construido y guardado en pgvector")

    return index, vector_store


def get_retriever(top_k: int = 6):
    """
    Retriever con MetadataReplacementPostProcessor:
    - Recupera top_k nodos por similitud (cada nodo = 1 frase)
    - MetadataReplacementPostProcessor sustituye el texto de cada nodo
      por su ventana (frase + vecinas) antes de pasarlo al LLM
    """
    base_retriever = _index.as_retriever(similarity_top_k=top_k)

    # Postprocesador: reemplaza frase por ventana
    window_postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )
    return base_retriever, window_postprocessor


# --- Generación directa con Ollama REST ---
def generate_answer_stream(query: str, nodes: list):
    # Filtramos nodos con score bajo
    MIN_SCORE = 0.65
    nodes = [n for n in nodes if n.score is None or n.score >= MIN_SCORE]

    if not nodes:
        def no_context():
            msg = "Esa información no figura en los textos que tengo disponibles."
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return no_context()

    # El texto ya fue reemplazado por la ventana por MetadataReplacementPostProcessor
    context_parts = []
    for node in nodes:
        source = node.metadata.get("file_name", "desconocido")
        text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        context_parts.append(f"[{source}]\n{text}")
    context = "\n\n".join(context_parts)

    prompt = (
        f"Contexto relevante:\n{context}\n\n"
        f"Pregunta del usuario: {query}\n\n"
        f"Responde en español con todos los detalles que aparezcan en el contexto: "
        f"nombres, fechas, lugares y hechos específicos. "
        f"Si el contexto no contiene información relevante, indícalo claramente.\n"
        f"Respuesta:"
    )

    def stream_generator():
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"num_predict": 600, "temperature": 0.3, "top_p": 0.9},
                },
                stream=True,
                timeout=180,
            )
            for line in resp.iter_lines():
                if line:
                    obj = json.loads(line)
                    token = obj.get("response", "")
                    if token:
                        for frase in FRASES_A_ELIMINAR:
                            token = token.replace(frase, "")
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if obj.get("done"):
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        break
        except Exception as e:
            yield f"data: {json.dumps({'token': f'Error: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

    return stream_generator()


# --- Endpoints Flask ---

@app.route("/health", methods=["GET"])
def health():
    n_rows = count_rows()
    return jsonify({
        "status": "ok",
        "model": LLM_MODEL,
        "nodes_in_pgvector": n_rows,
        "window_size": WINDOW_SIZE,
    })


@app.route("/rebuild", methods=["POST"])
def rebuild():
    """Fuerza reprocesar todos los documentos vaciando la tabla pgvector."""
    global _index, _vector_store
    import psycopg2
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE data_llamaindex_sw_chunks")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"No se pudo truncar tabla: {str(e)}"}), 500
    _index, _vector_store = build_index()
    return jsonify({"message": "Índice reconstruido", "nodes": count_rows()}), 200


@app.route("/chat", methods=["POST"])
def chat():
    """RAG con SentenceWindow + streaming SSE. Body: {"question": "..."}"""
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "La pregunta no puede estar vacía"}), 400
    try:
        retriever, window_postprocessor = get_retriever(top_k=6)
        nodes = retriever.retrieve(question)
        # Aplicar MetadataReplacementPostProcessor: sustituye frase por ventana
        nodes = window_postprocessor.postprocess_nodes(nodes)
        return Response(
            stream_with_context(generate_answer_stream(question, nodes)),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/similar-chunks", methods=["POST"])
def similar_chunks_api():
    """Debug: muestra los nodos con ventana expandida. Body: {"query": "...", "top_k": 6}"""
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 6)
    if not query:
        return jsonify({"error": "Falta 'query'"}), 400
    try:
        retriever, window_postprocessor = get_retriever(top_k=top_k)
        nodes = retriever.retrieve(query)
        nodes = window_postprocessor.postprocess_nodes(nodes)
        result = [
            {
                "score": float(n.score) if n.score is not None else None,
                "source": n.metadata.get("file_name", "desconocido"),
                "original_sentence": n.metadata.get("original_sentence", "")[:150],
                "window_text": n.get_content(metadata_mode=MetadataMode.NONE)[:400] + "..."
                    if len(n.get_content(metadata_mode=MetadataMode.NONE)) > 400
                    else n.get_content(metadata_mode=MetadataMode.NONE),
            }
            for n in nodes
        ]
        return jsonify({"nodes": result, "count": len(result)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Arranque ---
if __name__ == "__main__":
    print("Construyendo índice con SentenceWindowNodeParser...")
    _index, _vector_store = build_index()
    print(f"Modelo LLM: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}")
    print(f"SentenceWindowNodeParser: window_size={WINDOW_SIZE}")
    print("MetadataReplacementPostProcessor activo")
    app.run(host="0.0.0.0", port=5004, debug=False)
