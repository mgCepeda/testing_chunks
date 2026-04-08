"""
RAG API v3 - Chunking jerárquico con llama-index:
- HierarchicalNodeParser: divide en padres (~1024 chars) e hijos (~256 chars)
- AutoMergingRetriever: busca por hijos, sube al padre si hay suficientes coincidencias
- PGVectorStore: almacena embeddings de hijos en pgvector
- SimpleDocumentStore: guarda TODOS los nodos (padres e hijos) en disco (docstore_hier.json)
- Generación: llamada REST directa a Ollama (sin query_engine de llama-index → sin problema de prompt)
- Puerto: 5003
"""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()
from translation import translate_ca_to_en, translate_en_to_ca
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
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

# Fichero donde se guardan todos los nodos (padres + hijos) para no reprocesar
DOCSTORE_PATH = os.path.join(os.path.dirname(__file__), "docstore_hier.json")

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
_storage_context = None


def build_index():
    """
    Construye o carga el índice jerárquico:
    1. HierarchicalNodeParser → chunks en 2 niveles: padre=1024 chars, hijo=256 chars
    2. get_leaf_nodes() → solo los hijos van al VectorStoreIndex (pgvector)
    3. SimpleDocumentStore → guarda TODOS los nodos (incluidos padres sin embedding)
    4. Si ya existe docstore_hier.json, carga sin reprocesar documentos
    """
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )

    vector_store = PGVectorStore.from_params(
        **DB_CONFIG,
        table_name="llamaindex_hier_chunks",   # tabla propia, no interfiere con v2/v3
        embed_dim=EMBED_DIM,
    )

    if os.path.exists(DOCSTORE_PATH):
        print(f"Cargando docstore existente desde {DOCSTORE_PATH}...")
        docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        print(f"  - Nodos en docstore: {len(docstore.docs)}")
    else:
        print("Procesando documentos por primera vez...")
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        documents = [d for d in documents if "_en" in d.metadata.get("file_name", "")]
        print(f"  - Documentos en inglés cargados: {len(documents)}")

        # Dos niveles: padres de ~1024 chars, hijos de ~256 chars
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[1024, 256]
        )
        all_nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(all_nodes)

        print(f"  - Nodos totales: {len(all_nodes)}")
        print(f"  - Nodos hoja (level 1, con embedding): {len(leaf_nodes)}")

        # Todos los nodos (padres e hijos) van al docstore para que
        # AutoMergingRetriever pueda subir de hijo a padre
        docstore = SimpleDocumentStore()
        docstore.add_documents(all_nodes)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
        )

        # Solo los hijos se indexan en pgvector (son los que se buscan por similitud)
        index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        # Guardar docstore en disco para no reprocesar al reiniciar
        docstore.persist(DOCSTORE_PATH)
        print(f"  - Docstore guardado en {DOCSTORE_PATH}")

    return index, storage_context


def get_retriever(top_k: int = 6):
    """
    Crea el AutoMergingRetriever sobre el índice global.
    - Busca los top_k hijos más similares
    - Si ≥50% de los hijos de un padre coinciden → sube al padre (más contexto)
    """
    base_retriever = _index.as_retriever(similarity_top_k=top_k)
    return AutoMergingRetriever(
        base_retriever,
        _storage_context,
        simple_ratio_thresh=0.8,   # sube al padre solo si ≥80% de sus hijos coinciden
        verbose=True,
    )


# --- Generación directa con Ollama REST (sin llama-index query engine) ---
def generate_answer_stream(query: str, nodes: list):
    # Filtramos nodos con score bajo (ruido de otros libros)
    MIN_SCORE = 0.65
    nodes = [n for n in nodes if n.score is None or n.score >= MIN_SCORE]

    if not nodes:
        def no_context():
            msg = "Aquesta informació no figura en els textos que tinc disponibles."
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return no_context()

    # Construimos el contexto con el texto limpio (sin metadatos de llama-index)
    context_parts = []
    for i, node in enumerate(nodes):
        source = node.metadata.get("file_name", "desconocido")
        text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        context_parts.append(f"[{source}]\n{text}")
    context = "\n\n".join(context_parts)

    prompt = (
        f"Relevant context:\n{context}\n\n"
        f"User question: {query}\n\n"
        f"Answer in English with all details from the context: "
        f"names, dates, places and specific facts. "
        f"If the context does not contain relevant information, state it clearly.\n"
        f"Answer:"
    )

    def stream_generator():
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 600, "temperature": 0.3, "top_p": 0.9},
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
    n_docs = len(_storage_context.docstore.docs) if _storage_context else 0
    return jsonify({
        "status": "ok",
        "model": LLM_MODEL,
        "nodes_in_docstore": n_docs,
        "docstore_persisted": os.path.exists(DOCSTORE_PATH),
    })


@app.route("/rebuild", methods=["POST"])
def rebuild():
    """Fuerza reprocesar todos los documentos eliminando el docstore guardado."""
    global _index, _storage_context
    if os.path.exists(DOCSTORE_PATH):
        os.remove(DOCSTORE_PATH)
    _index, _storage_context = build_index()
    return jsonify({"message": "Índice reconstruido", "nodes": len(_storage_context.docstore.docs)}), 200


@app.route("/chat", methods=["POST"])
def chat():
    """RAG jerárquico con AutoMergingRetriever + streaming SSE. Body: {"question": "..."}"""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "La pregunta no puede estar vacía"}), 400

    try:
        question_en = translate_ca_to_en(question)
        retriever = get_retriever(top_k=6)
        nodes = retriever.retrieve(question_en)
        return Response(
            stream_with_context(generate_answer_stream(question_en, nodes)),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/similar-chunks", methods=["POST"])
def similar_chunks_api():
    """Debug: muestra los nodos recuperados (hijos o padres si hubo merge). Body: {"query": "...", "top_k": 6}"""
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 6)
    if not query:
        return jsonify({"error": "Falta 'query'"}), 400
    try:
        retriever = get_retriever(top_k=top_k)
        nodes = retriever.retrieve(query)
        result = [
            {
                "score": float(n.score) if n.score is not None else None,
                "source": n.metadata.get("file_name", "desconocido"),
                "text": n.get_content(metadata_mode=MetadataMode.NONE)[:300] + "..." if len(n.get_content(metadata_mode=MetadataMode.NONE)) > 300 else n.get_content(metadata_mode=MetadataMode.NONE),
            }
            for n in nodes
        ]
        return jsonify({"nodes": result, "count": len(result)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Arranque ---
if __name__ == "__main__":
    print("Construyendo índice jerárquico con llama-index...")
    _index, _storage_context = build_index()
    print(f"Modelo LLM: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}")
    print("HierarchicalNodeParser: chunk_sizes=[1024, 256]")
    print("AutoMergingRetriever activo")
    app.run(host="0.0.0.0", port=5003, debug=False)
