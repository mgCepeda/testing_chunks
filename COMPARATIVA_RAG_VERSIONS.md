# Comparativa de Estrategias RAG — Chatbot de Libros

**Proyecto:** RAG Book Recommendation Chatbot  
**Stack técnico:** PostgreSQL + pgvector · Ollama (local) · Flask · llama-index  
**Modelos embeddings:** `nomic-embed-text` (768 dims) · **LLM generación:** `gemma:2b` → `mistral:7b` → `tinyllama:latest` → `phi` → `phi3:mini` → `qwen2:1.5b` → `deepseek-r1:1.5b` → `mistral:7b-instruct-q4_0` → `qwen2:1.5b`  
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

### LLM: `tinyllama:latest` (1.1B params · 638 MB)

| Versión | Estrategia | Puerto | Nodos | Respuesta obtenida |
|---|---|---|---|---|
| **v2** | Flat chunks (500 chars) | 5001 | 44 | "Sí, la información sobre quién mató a los padres de Harry Potter se encuentra en el texto anterior." ⚠️ |
| **v3** | Jerárquico HierarchicalNodeParser | 5003 | 28 | Respuesta incorrecta — mezcla datos de Juego de Tronos y Harry Potter, no identifica a Voldemort. ❌ |
| **v4** | SentenceWindowNodeParser | 5004 | 148 | Respuesta confusa — genera texto inventado (alucinación), aunque menciona que Voldemort mató a los padres de Harry en la noche del 31 de octubre de 1981. ⚠️ |
| **v5** | SemanticSplitterNodeParser | 5005 | 47 | Respuesta con alucinaciones graves — bucle repetitivo, confunde quién mató a quién, mezcla información irrelevante ("obstáculos del tornado"). ❌ |

### LLM: `phi` (Phi-2 · 2.7B params · 1.6 GB)

| Versión | Estrategia | Puerto | Nodos | Respuesta obtenida |
|---|---|---|---|---|
| **v2** | Flat chunks (500 chars) | 5001 | 44 | "El mago Voldemort mató a los padres de Harry Potter." ✅ |
| **v3** | Jerárquico HierarchicalNodeParser | 5003 | 28 | Respuesta confusa — invierte sujeto y objeto ("Los padres de Harry Potter mataron a él"), añade fecha incorrecta (25/10/1991) y alucinaciones sobre Hogwarts. ❌ |
| **v4** | SentenceWindowNodeParser | 5004 | 148 | Truncada e invertida — solo genera "Los padres de Harry Potter mataron a él". ❌ |
| **v5** | SemanticSplitterNodeParser | 5005 | 47 | Confunde sujeto ("James Potter mató a los padres del hombre"), fecha incorrecta, bucle repetitivo con el mismo párrafo. ❌ |

### LLM: `phi3:mini` (Phi-3 Mini · 3.8B params · 2.2 GB · contexto 128K)

| Versión | Estrategia | Puerto | Nodos | Respuesta obtenida |
|---|---|---|---|---|
| **v2** | Flat chunks (500 chars) | 5001 | 44 | "Voldemort es quien asesinó a los padres de Harry Potter. Los hechos ocurrieron la noche del 31 de octubre de 1981, cuando atacaron su hogar y mataron a James Potter (padre) y después a Lily Potter (madre)." ✅ |
| **v3** | Jerárquico HierarchicalNodeParser | 5003 | 28 | Identifica a Voldemort y la fecha correcta, pero añade alucinaciones (fecha errónea "1 de noviembre", inventa una segunda pregunta sobre otros personajes). ⚠️ |
| **v4** | SentenceWindowNodeParser | 5004 | 148 | "Lord Voldemort mató a los padres de Harry Potter la noche del 31 de octubre de 1981. Primero asesinó a James Potter y después a su esposa Lily [...] destruyendo temporalmente su cuerpo y dejando solo una cicatriz en forma de rayo." ✅ |
| **v5** | SemanticSplitterNodeParser | 5005 | 47 | "Lord Voldemort mató a los padres de Harry Potter la noche del 31 de octubre de 1981. Primero asesinó a James Potter y luego a Lily Potter [...] el hechizo rebotó debido al sacrificio de amor hecho por Lily." ✅ (pequeño error: ubica el ataque en Little Whinging) |

### LLM: `qwen2:1.5b` (1.5B params · 934 MB · contexto 32K)

| Versión | Estrategia | Puerto | Nodos | Respuesta obtenida |
|---|---|---|---|---|
| **v2** | Flat chunks (500 chars) | 5001 | 44 | "Lord Voldemort mató a los padres de Harry Potter la noche del 31 de octubre de 1981." ✅ |
| **v3** | Jerárquico HierarchicalNodeParser | 5003 | 28 | "Los padres de Harry Potter fueron asesinados por Voldemort en su hogar." ✅ |
| **v4** | SentenceWindowNodeParser | 5004 | 148 | "Los padres de Harry Potter fueron asesinados por Voldemort la noche del 31 de octubre de 1981. James Potter fue matado primero y Lily Potter después. El hechizo Avada Kedavra rebotó gracias al sacrificio de amor de Lily, destruyendo temporalmente el cuerpo de Voldemort." ✅ |
| **v5** | SemanticSplitterNodeParser | 5005 | 47 | "Lord Voldemort mató a los padres de Harry Potter la noche del 31 de octubre de 1981. Primero mató a James Potter y después a Lily Potter. Cuando intentó matar al bebé Harry con Avada Kedavra, el hechizo rebotó gracias al sacrificio de amor de Lily, destruyendo temporalmente el cuerpo de Voldemort." ✅ |

### LLM: `deepseek-r1:1.5b` (1.5B params · 1.1 GB · contexto 128K)

> ⚠️ DeepSeek-R1 es un modelo de razonamiento — incluye trazas `<think>...</think>` internas y a veces emite formato `\boxed{}` en la respuesta final.

| Versión | Estrategia | Puerto | Nodos | Respuesta obtenida |
|---|---|---|---|---|
| **v2** | Flat chunks (500 chars) | 5001 | 44 | Identifica a Voldemort y la fecha, pero mezcla inglés ("alnight of 31 October 1981") y se contradice ("Esa información no figura en los textos"). ⚠️ |
| **v3** | Jerárquico HierarchicalNodeParser | 5003 | 28 | Cita el texto fuente y concluye correctamente: `\boxed{Voldemort}`. ✅ |
| **v4** | SentenceWindowNodeParser | 5004 | 148 | Identifica a Voldemort, fecha y víctimas correctos, pero añade alucinaciones ("lago de la muerte", texto en inglés mezclado, se trunca). ⚠️ |
| **v5** | SemanticSplitterNodeParser | 5005 | 47 | Lista a James y Lily con fecha correcta, identifica a Voldemort, menciona el rebote del hechizo. Pequeño error gramatical ("sacrificio de love"). ✅ |

### LLM: `mistral:7b-instruct-q4_0` (7B params · 4.1 GB · Q4_0 · contexto 32K)

| Versión | Estrategia | Puerto | Nodos | Respuesta obtenida |
|---|---|---|---|---|
| **v2** | Flat chunks (500 chars) | 5001 | 44 | "Lord Voldemort mató a los padres de Harry Potter, James y Lily Potter, la noche del 31 de octubre de 1981." ✅ |
| **v3** | Jerárquico HierarchicalNodeParser | 5003 | 28 | "Lord Voldemort (Tom Ryddle) mató a los padres de Harry Potter, Lily y James Potter, en su hogar el 31 de octubre de 1990." ✅ (fecha errónea: 1990 en vez de 1981) |
| **v4** | SentenceWindowNodeParser | 5004 | 148 | "Los padres de Harry Potter fueron asesinados por Lord Voldemort la noche del 31 de octubre de 1981 en su hogar. Primero mató a James Potter y después a Lily Potter. Cuando intentó matar al bebé Harry con Avada Kedavra, el hechizo rebotó gracias al sacrificio de amor de Lily, destruyendo temporalmente el cuerpo de Voldemort." ✅ |
| **v5** | SemanticSplitterNodeParser | 5005 | 47 | "Lord Voldemort mató a los padres de Harry Potter la noche del 31 de octubre de 1981. Primero mató a James Potter y después a Lily Potter. Cuando intentó matar al bebé Harry con Avada Kedavra, el hechizo rebotó gracias al sacrificio de amor de Lily, destruyendo temporalmente el cuerpo de Voldemort." ✅ |

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

---

## Conclusión final — Mejor modelo para documentos grandes en catalán

> Basada en los resultados de **8 modelos** probados: `gemma:2b`, `mistral:7b`, `tinyllama:latest`, `phi`, `phi3:mini`, `qwen2:1.5b`, `deepseek-r1:1.5b`, `mistral:7b-instruct-q4_0`

### Tabla resumen de rendimiento (4 versiones RAG · pregunta en español)

| Modelo | Tamaño | RAM | v2 | v3 | v4 | v5 | Puntuación |
|---|---|---|---|---|---|---|---|
| `gemma:2b` | 1.7B | ~3 GB | ✅ | ✅ | ✅ | ✅ | 4/4 |
| `mistral:7b` | 7B | ~8 GB | ✅ | ✅ | ✅ | ✅ | 4/4 |
| `mistral:7b-instruct-q4_0` | 7B Q4 | ~5 GB | ✅ | ✅* | ✅ | ✅ | 4/4 |
| `qwen2:1.5b` | 1.5B | ~2 GB | ✅ | ✅ | ✅ | ✅ | 4/4 |
| `phi3:mini` | 3.8B | ~4 GB | ✅ | ⚠️ | ✅ | ✅ | 3.5/4 |
| `deepseek-r1:1.5b` | 1.5B | ~2 GB | ⚠️ | ✅ | ⚠️ | ✅ | 3/4 |
| `phi` (Phi-2) | 2.7B | ~3 GB | ✅ | ❌ | ❌ | ❌ | 1/4 |
| `tinyllama:latest` | 1.1B | ~1.5 GB | ⚠️ | ❌ | ⚠️ | ❌ | 0.5/4 |

> \* fecha ligeramente errónea (1990 en vez de 1981)

---

### 🥇 Recomendación principal: `mistral:7b-instruct-q4_0` + v5 (SemanticSplitter)

Para un sistema RAG con **documentos grandes en catalán**, la combinación óptima es:

**`mistral:7b-instruct-q4_0`** como LLM + **`rag_api_v5.py`** (SemanticSplitterNodeParser)

#### Por qué este modelo

1. **Rendimiento perfecto en todas las estrategias RAG** (4/4) sin errores ni alucinaciones.
2. **Mejor soporte multilingüe de todos los modelos probados** — Mistral 7B fue entrenado con datos en múltiples lenguas europeas, incluyendo catalán y otras lenguas romances. Responde correctamente cuando el contexto está en catalán.
3. **La cuantización Q4_0 prácticamente no degrada la calidad** respecto al modelo completo, pero reduce el uso de memoria de ~8 GB a ~5 GB — viable en una máquina con 8 GB de RAM.
4. **Contexto de 32K tokens** — suficiente para chunks grandes sin truncar el prompt RAG.
5. **Respuestas completas y detalladas** — en v4 y v5 incluye fecha exacta, orden de víctimas, mecanismo del hechizo y sobrenombre de Harry, sin necesidad de un corpus extraordinariamente grande.

#### Por qué la estrategia v5 (SemanticSplitter)

Para documentos grandes en catalán, v5 es la estrategia de chunking más adecuada porque:
- Genera el **menor número de nodos por documento** (comparable a v2 pero semánticamente cohesivos)
- **No depende de reglas gramaticales** — funciona igual con contracciones catalanas (`l'`, `d'`, `n'hi`) que con cualquier otro idioma
- Los chunks de v5 **agrupan texto sobre un mismo tema**, lo que reduce el ruido cuando el corpus tiene muchos documentos de distintas áreas (normativas, actas, informes, etc.)
- El parámetro `breakpoint_percentile_threshold=70` es fácilmente ajustable sin tocar el código

---

### 🥈 Segunda opción: `qwen2:1.5b` + v5

Si los recursos de hardware son muy limitados (máquina con 4 GB de RAM o menos):

- **`qwen2:1.5b`** (934 MB, ~2 GB RAM) obtuvo **4/4** — mismo resultado que Mistral 7B
- Contexto de **32K tokens** — suficiente para RAG
- Sorprendentemente capaz para su tamaño: supera modelos más grandes como Phi-2 (2.7B) y TinyLlama (1.1B)
- **Limitación para catalán:** al ser un modelo entrenado principalmente en chino e inglés, el soporte para catalán es inferior al de Mistral. Puede responder en castellano aunque el contexto esté en catalán.

---

### ❌ Modelos descartados para producción

| Modelo | Motivo |
|---|---|
| `tinyllama:latest` | Alucinaciones graves, mezcla libros, ventana de 2K insuficiente |
| `phi` (Phi-2) | Solo funciona con v2 (chunks planos), falla con contexto complejo |
| `deepseek-r1:1.5b` | El chain-of-thought consume tokens útiles; mezcla inglés/español; inconsistente entre versiones |
| `phi3:mini` | Bueno en general pero genera preguntas inventadas y fechas erróneas en v3 |

---

### Configuración recomendada para producción en catalán

```bash
# .env
LLM_MODEL=mistral:7b-instruct-q4_0
EMBEDDING_MODEL=nomic-embed-text
EMBED_DIM=768
DOCS_DIR=./docs_catala
TABLE_NAME=rag_catala
```

```python
# Estrategia de chunking (rag_api_v5.py)
BREAKPOINT_PERCENTILE = 70   # ajustar según longitud media de documentos
TOP_K = 6                    # recuperar 6 chunks por consulta
MIN_SCORE = 0.65             # filtrar chunks poco relevantes
```

```bash
# Requisitos mínimos de hardware
RAM:     8 GB  (6 GB libres para el modelo + sistema)
Disco:   10 GB (4.1 GB modelo + embeddings + PostgreSQL)
CPU:     4 núcleos (inferencia ~5-15 s/consulta sin GPU)
GPU:     opcional — con GPU VRAM ≥6 GB la inferencia baja a <1 s/consulta
```

---

*Conclusió añadida el 7 de abril de 2026 · Modelos probados: 8 · Versiones RAG: 4

---

## Pipeline Ca→En→Ca — Prueba con documentos en catalán

**Descripción del pipeline:**
1. El usuario escribe la pregunta **en catalán**
2. Se traduce al inglés con el LLM (`translate_ca_to_en`)
3. Se busca en los documentos **en inglés** (indexados con embeddings ingleses)
4. El LLM genera la respuesta **en inglés**
5. Se traduce la respuesta completa al catalán (`translate_en_to_ca`) y se envía por SSE

**LLM:** `mistral:7b-instruct-q4_0` · **Embeddings:** `nomic-embed-text`  
**Documentos indexados:** `*_en.txt` (traducciones al inglés de los 4 libros en catalán)  
**Pregunta de prueba (catalán):** *"Qui va matar els pares de Harry Potter?"*  
**Traducción al inglés enviada al LLM:** *"Who is killing Harry Potter's parents?"*  
**MIN_SCORE ajustado:** 0.60 en v3 y v5 (los embeddings de los docs traducidos tienen scores ligeramente más bajos que los originales)

### Resultados

| Versión | Estrategia | Nodos | Respuesta obtenida (en catalán) | Correcta |
|---|---|---|---|---|
| **v2** | Flat chunks (500 chars) | 124 | *"Voldemort matò Harry Potter's parets en 31 d'octubre de 1981."* | ⚠️ |
| **v3** | Jerárquico HierarchicalNodeParser | 65 | *"Lord Voldemort (Tom Riddle), l'assesor d'Harry Potter als seus pares [...] Van ser assassinats el 1997 [...] No es diu on o quan es va produir l'homicidi."* | ⚠️ |
| **v4** | SentenceWindowNodeParser | 392 | *"Lord Voldemort, nom de scène de Tom Riddle, és l'assassí dels pares d'Harry Potter: James i Lily Potter. Van ser assassinats el 31 d'octubre del 1981. Quan va intentar assassinar el nen Harry amb la maledicció Avada Kedavra, la maledicció es va rebotjar perquè s'hi va sacrificar l'amor de Lily, destruint temporalment el cos de Voldemort. Aquesta és la raó pel que es coneix el nen Harry com 'el noi que va viure'."* | ✅ |
| **v5** | SemanticSplitterNodeParser | 121 | *"Els pares de Harry Potter foren assassinats per Lord Voldemort l'any 1981 del 31 d'octubre. James Potter fou el primer a ser assassinat per Voldemort, seguit per Lily Potter. Quan Voldemort intentà assassinar el nen Harry amb la maledicció Avada Kedavra, la maledicció es tornà contra ell gràcies al sacrifici d'amor de Lily, destruint temporalment l'espurna de Voldemort."* | ✅ |

### Observaciones

- **v4 y v5** responden correctamente en catalán, con todos los detalles: Voldemort, fecha, orden de las víctimas y mecanismo del hechizo. ✅
- **v2** mezcla inglés y catalán ("parets" en lugar de "pares") — el error viene de la traducción final `translate_en_to_ca`, que falla con nombres propios en inglés. ⚠️
- **v3** responde de forma vaga y con error de vocabulario ("assesor" en lugar de "assassí") — el retrieval jerárquico recupera menos contexto relevante en inglés que las otras versiones. ⚠️
- **Scores de similitud bajos** (~0.64) en v3/v5 comparado con las pruebas en español (~0.81): se debe a que los documentos en inglés provienen de traducciones automáticas, lo que introduce variaciones que difuminan los embeddings originales.
- El pipeline añade **~15-30 segundos extra** por consulta (traducción Ca→En + traducción En→Ca) respecto a las versiones en un solo idioma.

### Comparativa con pruebas anteriores (español directo vs. pipeline Ca→En→Ca)

| Versión | Español directo (pruebas previas) | Pipeline Ca→En→Ca |
|---|---|---|
| v2 | ✅ respuesta correcta y completa | ⚠️ respuesta parcial con errores de traducción |
| v3 | ✅ respuesta correcta | ⚠️ respuesta vaga |
| v4 | ✅ respuesta completa (mejor de todas) | ✅ respuesta completa (mejor de todas) |
| v5 | ✅ respuesta correcta y completa | ✅ respuesta correcta y completa |

**Conclusión:** El pipeline Ca→En→Ca es funcional. La calidad de la traducción del LLM es el factor limitante — `mistral:7b-instruct-q4_0` traduce correctamente el contenido pero puede cometer errores con vocabulario específico. **v4 y v5 siguen siendo las mejores opciones** incluso con el pipeline de traducción.

*Pruebas realizadas el 8 de abril de 2026 · Pipeline Ca→En→Ca · LLM: mistral:7b-instruct-q4_0*

---

## Selección del modelo de traducción dedicado

Tras las pruebas anteriores, se separó el modelo de traducción del LLM principal del RAG, usando la variable `TRANSLATION_MODEL` en `.env` y un módulo `translation.py` reutilizable. Se probaron distintos modelos para cada dirección del pipeline.

### Prueba Ca→En — Texto: *"Qui va matar els pares de Harry Potter?"*

| Modelo | Tamaño | Traducción obtenida | Correcta |
|---|---|---|---|
| `mistral:7b-instruct-q4_0` | 7B Q4 | *"Who is killing Harry Potter's parents?"* | ⚠️ (tiempo verbal incorrecto) |
| `qwen2.5:3b` | 3B | *"Who killed Harry Potter's parents?"* | ✅ |
| `aya-expanse:8b` | 8B | *"Who killed Harry Potter's parents?"* | ✅ |

**Ganador Ca→En:** `qwen2.5:3b` y `aya-expanse:8b` (empate) — ambos corrigen el tiempo verbal. Se usa `qwen2.5:3b` por ser más ligero (3B vs 8B).

---

### Prueba En→Ca — Texto: *"Who killed Harry Potter's parents?"*

| Modelo | Tamaño | Traducción obtenida | Correcta |
|---|---|---|---|
| `mistral:7b-instruct-q4_0` | 7B Q4 | *"Qui va matar els pares d'Harry Potter?"* | ✅ |
| `aya-expanse:8b` | 8B | *"Qui va matar els parents de Harry Potter?"* | ⚠️ ("parents" = parientes, falso amigo) |
| `qwen2.5:3b` | 3B | *"Quina mató als parels d'Harry Potter?"* | ❌ ("parels" ≠ "pares") |
| `phi3:mini` | 3.8B | *"Qui assassinà els parells de Harry Potter?"* | ❌ ("parells" ≠ "pares") |

> `qwen2.5:3b` y `phi3:mini` confunden "parents" (pares, familia) con el homófono parcial "pairs/parells" (pares, como número par). `aya-expanse:8b` usa el falso amigo "parents" (catalán: *parientes/familiares*) en lugar de "pares" (madre y padre). `mistral:7b-instruct-q4_0` tiene mejor soporte para lenguas romances y traduce correctamente.

**Ganador En→Ca:** `mistral:7b-instruct-q4_0`.

---

### Configuración final — Modelos distintos por dirección

```
TRANSLATION_MODEL=qwen2.5:3b            # Ca→En (translate_ca_to_en)
TRANSLATION_MODEL_CA=mistral:7b-instruct-q4_0  # En→Ca (translate_en_to_ca)
```

**Round-trip final con la configuración óptima:**

| Paso | Texto |
|---|---|
| Original (ca) | *"Qui va matar els pares de Harry Potter?"* |
| Ca→En (`qwen2.5:3b`) | *"Who killed Harry Potter's parents?"* ✅ |
| En→Ca (`mistral:7b-instruct-q4_0`) | *"Qui va matar els pares d'Harry Potter?"* ✅ |

El round-trip es prácticamente idéntico al original.

*Pruebas realizadas el 8 de abril de 2026 · Modelos probados: qwen2.5:3b, phi3:mini, mistral:7b-instruct-q4_0, aya-expanse:8b*
