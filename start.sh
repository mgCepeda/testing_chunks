#!/bin/bash

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Arrancando la aplicación ==="

# Terminal 1: Backend libros (puerto 5000)
echo "[1/3] Arrancando backend de libros (puerto 5000)..."
gnome-terminal --title="Backend Libros" -- bash -c "
  cd '$BASE_DIR/backend'
  source '$BASE_DIR/venv/bin/activate'
  FLASK_APP=flaskr FLASK_ENV=development flask run --port=5000
  exec bash
" 2>/dev/null || xterm -title "Backend Libros" -e "
  cd '$BASE_DIR/backend' &&
  source '$BASE_DIR/venv/bin/activate' &&
  FLASK_APP=flaskr FLASK_ENV=development flask run --port=5000;
  bash
" &

sleep 1

# Terminal 2: API RAG (puerto 5001)
echo "[2/3] Arrancando API RAG (puerto 5001)..."
gnome-terminal --title="API RAG" -- bash -c "
  cd '$BASE_DIR'
  source '$BASE_DIR/venv/bin/activate'
  python rag_api.py
  exec bash
" 2>/dev/null || xterm -title "API RAG" -e "
  cd '$BASE_DIR' &&
  source '$BASE_DIR/venv/bin/activate' &&
  python rag_api.py;
  bash
" &

sleep 1

# Terminal 3: Frontend React (puerto 3000)
echo "[3/3] Arrancando frontend React (puerto 3000)..."
gnome-terminal --title="Frontend React" -- bash -c "
  cd '$BASE_DIR/frontend'
  npm start
  exec bash
" 2>/dev/null || xterm -title "Frontend React" -e "
  cd '$BASE_DIR/frontend' &&
  npm start;
  bash
" &

echo ""
echo "=== Todo arrancado ==="
echo "  Frontend:      http://localhost:3000"
echo "  API libros:    http://localhost:5000"
echo "  API RAG:       http://localhost:5001"
