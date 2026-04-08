"""
translation.py
Módulo de traducción reutilizable para el pipeline Ca→En→Ca.

Funciones:
    translate_ca_to_en(text)  →  traduce catalán a inglés
    translate_en_to_ca(text)  →  traduce inglés a catalán

Usa TRANSLATION_MODEL de .env para las traducciones (separado del LLM_MODEL del RAG).
Si TRANSLATION_MODEL no está definido, usa LLM_MODEL como fallback.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b-instruct-q4_0")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", LLM_MODEL)
TRANSLATION_MODEL_CA = os.getenv("TRANSLATION_MODEL_CA", TRANSLATION_MODEL)


def translate_ca_to_en(text: str) -> str:
    """Traduce un texto del catalán al inglés."""
    prompt = (
        "Translate the following text from Catalan to English. "
        "Return only the translated text, with no explanation or commentary. "
        "Preserve proper nouns (character names, place names).\n\n"
        f"{text}"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": TRANSLATION_MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def translate_en_to_ca(text: str) -> str:
    """Traduce un texto del inglés al catalán."""
    prompt = (
        "Translate the following text from English to Catalan. "
        "Return only the translated text, with no explanation or commentary. "
        "Preserve proper nouns (character names, place names).\n\n"
        f"{text}"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": TRANSLATION_MODEL_CA, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()
