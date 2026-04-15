# Comparativa: Un solo modelo vs. Dos modelos especializados

**Motivación:** Evaluar si es viable usar un único modelo para las dos tareas — embeddings y generación de texto — para ahorrar RAM y simplificar el despliegue. Se prueban dos candidatos: `mistral:7b-instruct-q4_0` y `qwen2.5:3b`.

**Estrategia RAG:** v5 SemanticSplitterNodeParser · **Documentos:** `*_ca.txt` (catalán directo, sin traducción)  
**Pregunta de prueba:** *"Qui va matar els pares de Harry Potter?"*

---

## Configuraciones comparadas

| Configuración | Modelo embeddings | Dims | Modelo generación | Puerto | Tabla BD |
|---|---|---|---|---|---|
| **Dos modelos (referencia)** | `nomic-embed-text` | 768 | `mistral:7b-instruct-q4_0` | 5007 | `llamaindex_semantic_chunks_ca` |
| **Un solo modelo (mistral)** | `mistral:7b-instruct-q4_0` | 4096 | `mistral:7b-instruct-q4_0` | 5008 | `llamaindex_semantic_chunks_ca_mistral` |
| **Un solo modelo (qwen)** | `qwen2.5:3b` | 2048 | `qwen2.5:3b` | 5009 | `llamaindex_semantic_chunks_ca_qwen` |

---

## Rendimiento de indexación

| Métrica | Dos modelos (nomic) | Un solo modelo (mistral) | Un solo modelo (qwen) |
|---|---|---|---|
| Tiempo parsing (SemanticSplitter) | ~30 s | ~51 min | ~17 min |
| Tiempo indexación (embeddings → pgvector) | ~3 s | ~24 min | ~8 min |
| **Tiempo total** | **~33 s** | **~74 min** | **~24 min** |
| Nodos generados | 74 | 74 | 74 |
| Tamaño medio chunk | — | 327 chars | 327 chars |
| Tiempo por embedding | ~0.4 s | ~14 s | ~5 s |

> El SemanticSplitter necesita calcular embeddings para cada par de frases consecutivas para decidir dónde cortar. Con `nomic-embed-text` (modelo especializado, ~274 MB) es muy rápido. `qwen2.5:3b` (1.9 GB) es más ágil que `mistral` (4.1 GB) pero igualmente no está optimizado para esta tarea.

---

## Calidad del retrieval y la respuesta

| Configuración | Respuesta obtenida | Correcta |
|---|---|---|
| **Dos modelos (nomic + mistral)** | *"Lord Voldemort va matar els pares d'en Harry Potter la nit del 31 d'octubre de 1981. Primer va matar en James Potter (el pare d'en Harry) i després la Lily Potter (la mare d'en Harry). Quan va intentar matar el nadó Harry amb la maledicció Avada Kedavra, l'encanteri va rebotar gràcies al sacrifici d'amor de la Lily, destruint temporalment el cos de Voldemort. Per això en Harry és conegut com 'el nen que va sobreviure'."* | ✅ |
| **Un solo modelo (mistral)** | *"El context proporciona algunes referències a personatges i esdeveniments de la saga de Harry Potter, però no hi ha cap informació directament relativa al fet que els pares de Harry van ser assassinats. Per tant, no és possible respondre aquesta pregunta amb detalls específics en català basats en el context proporcionat."* | ❌ |
| **Un solo modelo (qwen)** | *"En la narrativa de la saga Harry Potter, és Voldemort qui va matar els pares d'en Harry Potter. [...] Per tant, no s'ha mencionat cap personatge que va ser responsable de la mort de les figures patents d'en Harry. La narració indica simplement que Voldemort és l'assassí..."* | ⚠️ |

### ¿Por qué fallan los modelos generativos como embedding model?

El problema es **arquitectural**: los dos objetivos de entrenamiento son incompatibles.

- **Modelos de embeddings especializados** (`nomic-embed-text`, `mxbai-embed-large`…) se entrenan con **aprendizaje contrastivo**: aprenden a que textos semánticamente similares tengan vectores cercanos y textos distintos tengan vectores lejanos. Su espacio vectorial refleja similitud de significado.
- **Modelos generativos** (`mistral`, `qwen`, `llama`…) se entrenan para **predecir el siguiente token**. Sus representaciones internas capturan información estadística del lenguaje, pero **no están calibradas para comparación de similitud coseno** entre frases arbitrarias.

Resultado práctico con `mistral`: el retriever devuelve chunks irrelevantes → el LLM responde "no tengo información". ❌  
Resultado práctico con `qwen2.5:3b`: el retriever recupera chunks parcialmente relevantes → la respuesta empieza correcta pero luego se contradice. ⚠️

No existe ningún modelo disponible en Ollama que sea excelente en ambas tareas. Los modelos "duales" como `E5-mistral-7b` o `GTE-Qwen2-7B` (generativos re-entrenados para embeddings con datos contrastivos) no están disponibles en Ollama, y al ser re-entrenados para embeddings pierden capacidad generativa.

---

## Resumen de recursos

| Configuración | RAM pico | Disco (modelos) | Modelos Ollama |
|---|---|---|---|
| Dos modelos (nomic + mistral) | ~5 GB (mistral dominante) | ~4.4 GB | 2 |
| Un solo modelo (mistral) | ~5 GB | ~4.1 GB | 1 |
| Un solo modelo (qwen) | ~2 GB | ~1.9 GB | 1 |

> El ahorro real de usar un solo modelo es mínimo en RAM (Ollama carga y descarga bajo demanda, no mantiene los dos cargados a la vez). Con qwen el ahorro en disco es mayor (~2.5 GB menos), pero al coste de un modelo generativo mucho menos capaz para catalán.

---

## Conclusión

> ❌ **Usar un modelo generativo como embedding model no es viable** para sistemas RAG.

Resumen de las tres configuraciones probadas:

| Configuración | Tiempo índice | Calidad retrieval | Veredicto |
|---|---|---|---|
| `nomic` + `mistral` (dos modelos) | ~33 s | ✅ Correcta y completa | **Recomendado** |
| `qwen2.5:3b` solo | ~24 min (×43) | ⚠️ Parcialmente correcta, contradictoria | ❌ No viable |
| `mistral` solo | ~74 min (×134) | ❌ No encuentra información | ❌ No viable |

La causa es arquitectural: los modelos generativos se entrenan para predecir el siguiente token, no para maximizar similitud coseno entre textos. Optimizar un objetivo degrada el otro — no existe un modelo único que sea excelente en ambas tareas (al menos no en Ollama).

`nomic-embed-text` es tan ligero (274 MB, ~500 MB RAM) que prácticamente no tiene coste añadido al sistema. La separación embeddings/generación es la arquitectura correcta para RAG local.

**Configuración recomendada:**

```
EMBEDDING_MODEL=nomic-embed-text    # 274 MB · especializado · rápido
LLM_MODEL=mistral:7b-instruct-q4_0 # 4.1 GB · generación en catalán
```

*Prueba realizada el 15 de abril de 2026 · Corpus: 4 libros en catalán · SemanticSplitterNodeParser (breakpoint=70)*
