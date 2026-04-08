"""
translate_docs.py
Traduce los documentos *_ca.txt (catalán) a inglés usando Ollama.
Guarda los resultados como *_en.txt en el mismo directorio.

Uso:
    venv/bin/python translate_docs.py
"""

import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b-instruct-q4_0")
DOCS_DIR = Path(__file__).parent / "rag_books" / "docs"


def translate_block(text: str) -> str:
    """Traduce un bloque de texto del catalán al inglés usando Ollama."""
    prompt = (
        "Translate the following text from Catalan to English. "
        "Return only the translated text, with no explanation, no preamble, "
        "no commentary. Preserve paragraph breaks and formatting.\n\n"
        f"{text}"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def translate_file(ca_path: Path) -> Path:
    stem = ca_path.stem.replace("_ca", "")
    en_path = ca_path.parent / f"{stem}_en.txt"

    print(f"\n📄 {ca_path.name}  →  {en_path.name}")

    text = ca_path.read_text(encoding="utf-8")

    # Dividir por bloques separados por línea vacía (párrafos / capítulos)
    blocks = text.split("\n\n")
    translated_blocks = []

    for i, block in enumerate(blocks):
        if not block.strip():
            translated_blocks.append("")
            continue
        print(f"   bloque {i + 1}/{len(blocks)}...", end=" ", flush=True)
        translated = translate_block(block.strip())
        translated_blocks.append(translated)
        print("✓")
        time.sleep(0.2)  # pequeña pausa entre llamadas

    en_text = "\n\n".join(translated_blocks)
    en_path.write_text(en_text, encoding="utf-8")
    print(f"   Guardado: {en_path}")
    return en_path


def main():
    ca_files = sorted(DOCS_DIR.glob("*_ca.txt"))
    if not ca_files:
        print("No se encontraron archivos *_ca.txt en", DOCS_DIR)
        return

    print(f"Modelo de traducción: {LLM_MODEL}")
    print(f"Archivos a traducir: {len(ca_files)}")

    for ca_path in ca_files:
        translate_file(ca_path)

    print("\n✅ Traducción completada.")


if __name__ == "__main__":
    main()
