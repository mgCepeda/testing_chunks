# Comparativa de Estrategias RAG — Chatbot de Libros

**Proyecto:** RAG Book Recommendation Chatbot  
**Stack técnico:** PostgreSQL + pgvector · Ollama (local) · Flask · llama-index  
**Modelos embeddings:** `nomic-embed-text` (768 dims) · **LLM generación:** `gemma:2b` → `mistral:7b`  
**Pregunta de prueba:** _"¿Quién mató a los padres de Harry Potter?"_

---

## Resumen ejecutivo

Se han desarrollado y comparado **5 versiones** del mismo sistema RAG, variando únicamente la estrategia de chunking (división de documentos en fragmentos). El resto del sistema (base de datos, embeddings, LLM) es idéntico en todas las versiones.

### LLM: `gemma:2b`

| Versión | Estrategia | Puerto | Nodos | Respuesta obtenida |
|---|---|---|---|---|
| **v2** | Flat chunks (500 chars) | 5001 | 44 | "El texto indica que Lord Voldemort mató a los padres de Harry Potter la noche del 31 de octubre de 1981." |
| **v3** | Jerárquico HierarchicalNodeParser | 5003 | 28 | "El personaje que mató a los padres de Harry Potter fue Voldemort (Tom Riddle)." |
| **v4** | SentenceWindowNodeParser | 5004 | 148 | "Los padres de Harry Potter fueron matados por Lord Voldemort la noche del 31 de octubre de 1981. Primero mató a James Potter (el padre) y después a Lily Potter (la madre)." |
| **v5** | SemanticSplitterNodeParser | 5005 | 47 | "Lord Voldemort mató a los padres de Harry Potter la noche del 31 de octubre de 1981." |

### LLM: `mistral:7b`

| Versión | Estrategia | Puerto | Nodos | Respuesta obtenida |
|---|---|---|---|---|
| **v2** | Flat chunks (500 chars) | 5001 | 44 | "Lord Voldemort mató a los padres de Harry Potter la noche del 31 de octubre de 1981." |
| **v3** | Jerárquico HierarchicalNodeParser | 5002 | 28 | "Lord Voldemort mató a los padres de Harry Potter." |
| **v4** | SentenceWindowNodeParser | 5004 | 148 | "Los padres de Harry Potter fueron asesinados por Lord Voldemort la noche del 31 de octubre de 1981 en su hogar. Primero mató a James Potter (el padre de Harry) y después a Lily Potter (la madre de Harry). Cuando intentó matar al bebé Harry con la maldición Avada Kedavra, el hechizo rebotó gracias al sacrificio de amor de Lily, destruyendo temporalmente el cuerpo de Voldemort. Por eso Harry es conocido como «el niño que sobrevivió»." |
| **v5** | SemanticSplitterNodeParser | 5005 | 47 | "Lord Voldemort mató a los padres de Harry Potter la noche del 31 de octubre de 1981 en su casa. Primero mató a James Potter (el padre de Harry) y después a Lily Potter (la madre de Harry). Cuando intentó matar al bebé Harry con la maldición Avada Kedavra, el hechizo rebotó gracias al sacrificio de amor de Lily, destruyendo temporalmente el cuerpo de Voldemort." |

---

## Descripción detallada de cada versión

### v2 — Flat Chunks (línea base)

**Cómo funciona:**
```
Texto original:  [----500 chars----][----500 chars----][----500 chars----]
                   corta cada 500 caracteres, con solapamiento de 100 chars
```

- Divide el texto cada 500 caracteres con solapamiento de 100
- Cada chunk recibe su propio embedding y se guarda en pgvector
- Recupera los top-K chunks más similares a la pregunta

**Ventajas:**
- Simple y predecible
- Fácil de depurar (todos los chunks tienen el mismo tamaño)
- Rápido de procesar

**Inconvenientes:**
- Corta en medio de frases o párrafos (pierde coherencia)
- El tamaño fijo ignora la estructura semántica del documento
- Con muchos documentos, chunks de distintos libros compiten con scores similares

---

### v3 — Chunking Jerárquico (HierarchicalNodeParser + AutoMergingRetriever)

**Cómo funciona:**
```
Nivel padre (1024 chars):  [==========PADRE 1==========][==========PADRE 2==========]
Nivel hijo  ( 256 chars):  [H1][H2][H3][H4]             [H5][H6][H7][H8]
                              ↑ se indexan en pgvector
                              si ≥80% de los hijos de un padre coinciden
                              → AutoMergingRetriever sube al PADRE (más contexto)
```

- Los hijos (256 chars) se indexan para retrieval preciso
- El AutoMergingRetriever puede "subir" al padre (1024 chars) si hay suficientes coincidencias
- Requiere un fichero `docstore_hier.json` en disco para guardar la relación padre-hijo

**Ventajas:**
- Equilibrio entre precisión (hijos pequeños) y contexto (padres grandes)
- Mecanismo inteligente de fusión de chunks relacionados
- Pocos nodos en pgvector (28 para 4 libros)

**Inconvenientes:**
- Más complejo de configurar (`simple_ratio_thresh`, tamaños de chunk)
- El fichero docstore en disco puede desincronizarse si se modifican documentos
- Chunks siguen siendo de tamaño fijo (ignora cambios de tema)

---

### v4 — SentenceWindowNodeParser + MetadataReplacementPostProcessor

**Cómo funciona:**
```
Frases:  [F1] [F2] [F3] [F4] [F5] [F6] [F7] [F8] [F9]

Nodo indexado (embedding):  [F4]  ← solo la frase, muy preciso
Metadatos del nodo:         [F2][F3][F4][F5][F6]  ← ventana de 3 frases vecinas

Al recuperar → MetadataReplacementPostProcessor sustituye F4 por la ventana completa
El LLM recibe:              [F2][F3][F4][F5][F6]  ← más contexto
```

- Embedding de una frase sola → máxima precisión semántica
- La "ventana" de frases vecinas (configurable, por defecto ±3) viaja en los metadatos
- No necesita fichero externo en disco (la ventana está dentro del nodo en pgvector)

**Ventajas:**
- **La que mejor responde** en las pruebas (más detalle, nombres propios, fechas)
- Sin ficheros externos de estado
- Sencilla de entender conceptualmente

**Inconvenientes:**
- Genera **muchos más nodos** (148 para 4 libros → escala mal con corpus grandes)
- El splitter de frases puede fallar con idiomas con muchas contracciones (`l'`, `d'`, `n'` en catalán)
- Más lento en retrieval con corpus grandes (más vectores que comparar)

---

### v5 — SemanticSplitterNodeParser

**Cómo funciona:**
```
Texto original:  "Sinopsis del libro. Harry era huérfano. Vivía en casa de los Dursley.
                  Un día llegó una carta. Era de Hogwarts. La escuela era mágica.
                  Voldemort mató a sus padres. Lo hizo la noche de Halloween."

Paso 1: El modelo de embeddings calcula la similitud entre cada par de frases consecutivas
Paso 2: Cuando la similitud baja del percentil 70 → corte (cambio de tema)
Resultado:
  Chunk 1: [Sinopsis del libro. Harry era huérfano. Vivía en casa de los Dursley.]
  Chunk 2: [Un día llegó una carta. Era de Hogwarts. La escuela era mágica.]
  Chunk 3: [Voldemort mató a sus padres. Lo hizo la noche de Halloween.]
```

- Usa los propios embeddings para detectar **cambios de tema** en el texto
- Cada chunk contiene texto sobre **un único tema** (chunks semánticamente cohesivos)
- Tamaño variable: un chunk puede ser 100 chars o 800 chars según la duración del tema

**Ventajas:**
- Genera el **menor número de nodos** comparado con v4 (47 vs 148 para el mismo corpus)
- Chunks más cohesivos → menos ruido cuando hay muchos documentos de distintos temas
- El `breakpoint_percentile` permite ajustar la granularidad según el corpus
- **Funciona bien en cualquier idioma** (la similitud semántica es language-agnostic)

**Inconvenientes:**
- El **primer procesado es más lento** (calcula embeddings dos veces: para el split y para la indexación)
- Requiere un embedding model multilingüe de buena calidad
- Más difícil de inspeccionar (los chunks tienen tamaños impredecibles)

---

## Comparativa de retrieval

Scores de similitud recuperados para la pregunta de prueba:

| Versión | Score #1 | Fuente #1 | Score #2 | Fuente #2 |
|---|---|---|---|---|
| v2 | 0.89 | harry_potter.txt ✅ | 0.71 | harry_potter.txt |
| v3 | 0.66 | harry_potter.txt ✅ | 0.65 | juego_de_tronos.txt ⚠️ |
| v4 | 0.82 | harry_potter.txt ✅ | 0.78 | harry_potter.txt |
| v5 | 0.81 | harry_potter.txt ✅ | 0.69 | harry_potter.txt |

> ⚠️ V3 recuperó un chunk de _Juego de Tronos_ con score casi idéntico al relevante — problema mitigado añadiendo un filtro de score mínimo (≥0.65) en producción.

---

## Recomendación para documentos en catalán (corpus grande)

### 🥇 V5 (SemanticSplitter) — **Recomendación principal**

Para un corpus de **muchos documentos en catalán**, la versión 5 es la más adecuada por las siguientes razones:

#### 1. Escalabilidad
Con 4 libros, los nodos generados son:

| Versión | Nodos (4 libros) | Estimación (100 docs) |
|---|---|---|
| v4 SentenceWindow | 148 | ~3.700 nodos |
| v2 Flat | 44 | ~1.100 nodos |
| **v5 Semantic** | **47** | **~1.175 nodos** |
| v3 Jerárquico | 28 (hijos) | ~700 nodos |

V5 genera un número de nodos comparable a v2 pero con **mucha mejor calidad semántica**.

#### 2. Compatibilidad con catalán
El modelo de embeddings `nomic-embed-text` es multilingüe y tiene buen soporte para lenguas romances, incluyendo catalán. La clave del SemanticSplitter es que opera **únicamente con vectores de similitud**, sin depender de reglas lingüísticas (puntuación, separadores de frases, etc.). Esto es crucial porque:

- Las contracciones catalanas (`l'home`, `d'aquí`, `n'hi ha`) podrían romper el splitter de frases de v4
- El SemanticSplitter no usa reglas gramaticales → funciona igual en catalán que en castellano o inglés

#### 3. Calidad de los chunks
Cada chunk de v5 agrupa oraciones que **hablan del mismo tema**. En documentos técnicos, legales o administrativos en catalán (por ejemplo, normativas, actas, informes), esto es especialmente valioso porque:

- Una normativa puede hablar de "definiciones", "ámbito de aplicación", "sanciones" en secciones distintas
- V5 separará automáticamente estos temas en chunks distintos
- V2 los mezclaría si caen dentro del mismo bloque de 500 chars

#### 4. Parámetro ajustable
El parámetro `breakpoint_percentile_threshold` permite ajustar sin reprogramar:

```python
# Documentos cortos o muy estructurados (formularios, fichas):
BREAKPOINT_PERCENTILE = 85  # chunks más grandes, menos cortes

# Documentos largos con muchos temas (informes, libros):
BREAKPOINT_PERCENTILE = 60  # chunks más pequeños, más cortes

# Valor por defecto recomendado para documentos administrativos:
BREAKPOINT_PERCENTILE = 70
```

#### 5. Coste de procesado vs beneficio
- **Solo se procesa una vez** por documento — el resultado queda guardado en pgvector
- Las **consultas** son igual de rápidas en todas las versiones (búsqueda vectorial en pgvector)
- El coste extra de procesado inicial (embeddings dobles) se paga una sola vez

---

### 🥈 V4 (SentenceWindow) — Segunda opción si el corpus es pequeño

Si el corpus tiene **menos de 50 documentos**, v4 es preferible porque:
- Las respuestas son más detalladas (la ventana de frases vecinas aporta más contexto al LLM)
- El número de nodos sigue siendo manejable

**Limitación en catalán:** el splitter de frases puede tener problemas con contracciones. Se puede mitigar pasando `sentence_splitter` personalizado.

---

### Sobre el modelo LLM para catalán

`gemma:2b` (el modelo actual) tiene soporte limitado para catalán — puede responder en castellano aunque el contexto sea en catalán. Para producción en catalán se recomienda:

| Modelo | Tamaño | Soporte catalán | RAM necesaria |
|---|---|---|---|
| `gemma:2b` | 1.7B params | Limitado | 4 GB |
| `mistral:7b` | 7B params | Bueno | 8 GB |
| `llama3:8b` | 8B params | Bueno | 10 GB |
| `llama3.1:70b` | 70B params | Excelente | 48 GB |

Para cambiar de modelo basta con:
```bash
LLM_MODEL=mistral:7b python rag_api_v5.py
```

---

## Arquitectura recomendada para producción

```
Documentos catalán (.txt, .pdf, .docx)
         │
         ▼
SimpleDirectoryReader (llama-index)
         │
         ▼
SemanticSplitterNodeParser
  breakpoint_percentile=70
  embed_model=nomic-embed-text
         │
         ▼
PGVectorStore (PostgreSQL + pgvector)
  tabla: chunks_catala
  dim: 768
         │
    consulta usuario
         │
         ▼
VectorStoreIndex.as_retriever(top_k=6)
  filtro: score >= 0.65
         │
         ▼
Ollama REST → mistral:7b o llama3:8b
  prompt en catalán
         │
         ▼
Respuesta en catalán (streaming SSE)
```

---

*Documento generado el 26 de marzo de 2026*  
*Entorno: Linux · Python 3.x · llama-index · PostgreSQL 16 · Ollama local*
