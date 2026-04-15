# Comparativa: Catalán directo vs. Pipeline Ca→En→Ca

**Objetivo:** Determinar si vale la pena usar el pipeline de traducción (Ca→En→Ca) o si es mejor indexar los documentos directamente en catalán y responder sin traducción.

**LLM:** `mistral:7b-instruct-q4_0` · **Embeddings:** `nomic-embed-text` (768 dims)  
**Pregunta de prueba:** *"Qui va matar els pares de Harry Potter?"*  
**Versiones comparadas:** v4 (SentenceWindow) y v5 (SemanticSplitter) — las mejor valoradas en pruebas anteriores

---

## Configuraciones comparadas

| Configuración | Docs indexados | Idioma pregunta | Idioma respuesta | Pipeline traducción | Puerto |
|---|---|---|---|---|---|
| **v4 pipeline** | `*_en.txt` (inglés) | catalán → inglés | inglés → catalán | Sí (qwen2.5:3b + mistral) | 5004 |
| **v4_ca directo** | `*_ca.txt` (catalán) | catalán | catalán | No | 5006 |
| **v5 pipeline** | `*_en.txt` (inglés) | catalán → inglés | inglés → catalán | Sí (qwen2.5:3b + mistral) | 5005 |
| **v5_ca directo** | `*_ca.txt` (catalán) | catalán | catalán | No | 5007 |

---

## Resultados

### v4 — SentenceWindowNodeParser

**Nodos indexados:**
- v4 pipeline (docs en inglés): 392 nodos
- v4_ca directo (docs en catalán): 235 nodos

| Configuración | Respuesta obtenida | Valoración |
|---|---|---|
| **v4 pipeline** | *"Lord Voldemort, nom de scène de Tom Riddle, és l'assassí dels pares d'Harry Potter: James i Lily Potter. Van ser assassinats el 31 d'octubre del 1981. Quan va intentar assassinar el nen Harry amb la maledicció Avada Kedavra, la maledicció es va rebotjar perquè s'hi va sacrificar l'amor de Lily, destruint temporalment el cos de Voldemort. Aquesta és la raó pel que es coneix el nen Harry com 'el noi que va viure'."* | ✅ |
| **v4_ca directo** | *"El personatge que va matar els pares de Harry Potter és Lord Voldemort. En la nit del 31 d'octubre de 1981, Lord Voldemort va matar en James Potter (el pare d'en Harry) i després la Lily Potter (la mare d'en Harry). Quan va intentar matar el nadó Harry amb la maledicció Avada Kedavra, l'encanteri va rebotar gràcies al sacrifici d'amor de la Lily, destruint temporalment el cos de Voldemort."* | ✅ |

### v5 — SemanticSplitterNodeParser

> ⏳ Pendiente de prueba — v5_ca en construcción

| Configuración | Respuesta obtenida | Valoración |
|---|---|---|
| **v5 pipeline** | *"Els pares de Harry Potter foren assassinats per Lord Voldemort l'any 1981 del 31 d'octubre. James Potter fou el primer a ser assassinat per Voldemort, seguit per Lily Potter. Quan Voldemort intentà assassinar el nen Harry amb la maledicció Avada Kedavra, la maledicció es tornà contra ell gràcies al sacrifici d'amor de Lily, destruint temporalment l'espurna de Voldemort."* | ✅ |
| **v5_ca directo** | *"Lord Voldemort va matar els pares d'en Harry Potter la nit del 31 d'octubre de 1981. Primer va matar en James Potter (el pare d'en Harry) i després la Lily Potter (la mare d'en Harry). Quan va intentar matar el nadó Harry amb la maledicció Avada Kedavra, l'encanteri va rebotar gràcies al sacrifici d'amor de la Lily, destruint temporalment el cos de Voldemort. Per això en Harry és conegut com 'el nen que va sobreviure'."* | ✅ |

---

## Análisis comparativo v4

### Calidad de la respuesta
Ambas respuestas son correctas e incluyen todos los detalles clave:
- ✅ Identifica a Lord Voldemort como asesino
- ✅ Fecha correcta: 31 de octubre de 1981
- ✅ Orden de las víctimas: James Potter primero, Lily Potter después
- ✅ Mecanismo del hechizo: Avada Kedavra rebota por el sacrificio de amor de Lily

### Diferencias observadas

| Aspecto | v4 pipeline | v4_ca directo |
|---|---|---|
| Naturalidad del catalán | Correcto pero algo formal (*"nom de scène"*, galicismo) | Más natural, usa artícles personals (`en James`, `la Lily`) |
| Nodos generados | 392 (docs en inglés, más frases) | 235 (docs originales en catalán) |
| Latencia extra | +15-30 s (2 llamadas de traducción) | Sin overhead |
| Precisión retrieval | Scores ~0.64 (docs traducidos) | Scores mayores esperados (docs originales) |
| Complejidad del sistema | Alta (módulo traducción, 2 modelos extra) | Baja (solo LLM + embeddings) |

---

## Conclusión parcial (v4)

> El **catalán directo es superior** en todos los aspectos: misma calidad de respuesta, catalán más natural, menos nodos, sin latencia de traducción y sistema más simple.

El pipeline Ca→En→Ca solo tiene sentido si:
1. Los documentos **ya existen en inglés** y no hay versión en catalán
2. El LLM elegido tiene **mal soporte para catalán** (no es el caso de `mistral:7b-instruct-q4_0`)
3. Se necesita **compatibilidad con corpus mixtos** (algunos docs en inglés, otros en catalán)

---

## Análisis comparativo v5

### Calidad de la respuesta
Ambas respuestas son correctas e incluyen todos los detalles:
- ✅ Identifica a Lord Voldemort
- ✅ Fecha correcta: 31 de octubre de 1981
- ✅ Orden de las víctimas: James primero, Lily después
- ✅ Mecanismo del hechizo: Avada Kedavra rebota por el sacrificio de amor de Lily
- ✅ v5_ca añade además: *"Per això en Harry és conegut com 'el nen que va sobreviure'"* — detalle extra ausente en v5 pipeline

### Diferencias observadas

| Aspecto | v5 pipeline | v5_ca directo |
|---|---|---|
| Naturalidad del catalán | Correcto pero arcaizante (*"foren assassinats"*, *"l'espurna"*) | Más coloquial y natural (*"va matar"*, *"el nen que va sobreviure"*) |
| Nodos generados | 121 (docs en inglés) | 74 (docs originales en catalán) |
| Latencia extra | +15-30 s (2 llamadas de traducción) | Sin overhead |
| Detalle extra | No menciona el sobrenombre de Harry | ✅ incluye "el nen que va sobreviure" |

---

## Conclusión final

| Configuración | Nodos | Calidad | Catalán natural | Latencia extra | Complejidad |
|---|---|---|---|---|---|
| v4 pipeline | 392 | ✅ | ⚠️ (galicismos) | +15-30 s | Alta |
| **v4_ca directo** | **235** | **✅** | **✅** | **0 s** | **Baja** |
| v5 pipeline | 121 | ✅ | ⚠️ (arcaizante) | +15-30 s | Alta |
| **v5_ca directo** | **74** | **✅+** | **✅** | **0 s** | **Baja** |

### 🥇 Recomendación: `rag_api_v5_ca.py` (SemanticSplitter · catalán directo)

**El pipeline Ca→En→Ca no aporta ninguna ventaja** cuando:
1. Los documentos **ya están en catalán**
2. El LLM (`mistral:7b-instruct-q4_0`) tiene **buen soporte nativo para catalán**

El catalán directo es estrictamente mejor: misma o mayor calidad de respuesta, catalán más natural, menos nodos en pgvector, sin latencia de traducción y sistema más simple de mantener.

**El pipeline solo justifica su complejidad si:**
- Los documentos originales están en inglés (no hay versión en catalán)
- El LLM elegido tiene soporte pobre para catalán

*Pruebas completadas el 15 de abril de 2026 · LLM: mistral:7b-instruct-q4_0 · Embeddings: nomic-embed-text*
