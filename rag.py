from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2

# Configurar el LLM y los embeddings con Ollama (todo en local)
Settings.llm = Ollama(model="gemma:2b", request_timeout=120.0, system_prompt=(
    "Eres un crítico literario de renombre con más de 30 años de experiencia. "
    "Tu pasión por la literatura es contagiosa: usas un tono culto pero cercano, con metáforas evocadoras y referencias precisas. "
    "Cuando hablas de un libro, describes su atmósfera, su importancia histórica y su impacto en los lectores. "
    "Siempre estructuras tus respuestas: primero el dato concreto, luego el análisis o contexto. "
    "Nunca inventas información: si no la tienes en el contexto, lo dices con elegancia."
))
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.chunk_size = 512
Settings.chunk_overlap = 64

# Configurar pgvector como base de datos vectorial
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "rag_books"
DB_USER = "marina"
DB_PASSWORD = "marina"
TABLE_NAME = "rag_embeddings"
EMBED_DIM = 768  # dimensión de nomic-embed-text

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
cur.execute(f"SELECT COUNT(*) FROM data_{TABLE_NAME}") if table_exists else None
row_count = cur.fetchone()[0] if table_exists else 0
cur.close()
conn.close()

if table_exists and row_count > 0:
    print(f"Cargando índice existente desde pgvector ({row_count} fragmentos)...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
else:
    # Cargar los documentos desde la carpeta docs y crear el índice
    docs_path = "../rag_books/docs"
    documents = SimpleDirectoryReader(docs_path).load_data()
    print(f"Documentos cargados: {len(documents)}")
    print("Generando embeddings y guardando en pgvector...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("Embeddings guardados en PostgreSQL.")

# Crear el motor de consultas (similarity_top_k=6 para recuperar más fragmentos)
query_engine = index.as_query_engine(similarity_top_k=4, response_mode="compact")

# Prompt personalizado para mejorar las respuestas
qa_prompt = PromptTemplate(
    "Eres un crítico literario apasionado. Usa ÚNICAMENTE la información del contexto para responder.\n"
    "IMPORTANTE: Nunca menciones el 'contexto', el 'texto proporcionado' ni frases como 'según el contexto' o 'basándome en'. Habla directamente como experto.\n"
    "Responde con entusiasmo y profundidad: menciona títulos, autores, fechas, personajes y detalles concretos.\n"
    "Añade análisis literario: habla de la importancia de la obra, su estilo narrativo o su impacto cultural cuando el contexto lo permita.\n"
    "Usa un lenguaje rico y evocador, como si estuvieras explicando a un apasionado de la literatura.\n"
    "Si hay libros pendientes de publicar, indícalo claramente.\n"
    "Si la información no está en el contexto, responde con elegancia: 'Esa información no figura en los textos que tengo disponibles'.\n\n"
    "Contexto:\n{context_str}\n\n"
    "Pregunta: {query_str}\n\n"
    "Respuesta del experto literario:"
)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

# Bucle interactivo de preguntas
print("\n=== RAG de Libros ===")
print("Escribe tu pregunta sobre los libros (o 'salir' para terminar)\n")

while True:
    question = input("Tu pregunta: ").strip()
    if question.lower() in ("salir", "exit", "quit"):
        break
    if not question:
        continue

    response = query_engine.query(question)
    text = str(response)
    # Eliminar frases que delatan el uso de contexto
    frases = [
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
    for frase in frases:
        text = text.replace(frase, "")
    # Capitalizar la primera letra si quedó en minúscula
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    print(f"\nRespuesta: {text}\n")
