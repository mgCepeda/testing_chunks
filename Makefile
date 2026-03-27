BASE_DIR := $(shell pwd)
VENV := $(BASE_DIR)/venv/bin/activate

.PHONY: all backend rag rag2 rag-only frontend stop postgres status

all:
	@echo "Arrancando todos los servicios..."
	@make postgres
	@make backend &
	@make rag &
	@make frontend

postgres:
	@echo "[PostgreSQL] Verificando servicio..."
	@pg_lsclusters | grep -q "online" && echo "[PostgreSQL] Ya está corriendo." || (sudo service postgresql start && echo "[PostgreSQL] Arrancado.")

backend:
	@echo "[Backend libros] Puerto 5000..."
	@lsof -ti:5000 > /dev/null 2>&1 && echo "[Backend] Ya está corriendo en puerto 5000." || \
		(cd $(BASE_DIR)/backend && source $(VENV) && FLASK_APP=flaskr FLASK_ENV=development flask run --port=5000)

rag:
	@echo "[API RAG] Puerto 5001..."
	@lsof -ti:5001 > /dev/null 2>&1 && echo "[API RAG] Ya está corriendo en puerto 5001." || \
		(cd $(BASE_DIR) && source $(VENV) && python rag_api.py)

frontend:
	@echo "[Frontend React] Puerto 3000..."
	@lsof -ti:3000 > /dev/null 2>&1 && echo "[Frontend] Ya está corriendo en puerto 3000." || \
		(cd $(BASE_DIR)/frontend && npm start)

stop:
	@echo "Parando todos los servicios..."
	@pkill -f "flask run --port=5000" || true
	@pkill -f "rag_api.py" || true
	@pkill -f "react-scripts/scripts/start" || true
	@echo "Servicios detenidos (PostgreSQL sigue corriendo)."

rag-only:
	@echo "[API RAG] Arrancando solo la API RAG en puerto 5001..."
	@make postgres
	@lsof -ti:5001 > /dev/null 2>&1 && echo "[API RAG] Ya está corriendo en puerto 5001." || \
		(cd $(BASE_DIR) && source $(VENV) && python rag_api.py)

rag2:
	@echo "[API RAG v2] Arrancando rag_api_v2 en puerto 5001 (arquitectura poc-ai)..."
	@make postgres
	@lsof -ti:5001 > /dev/null 2>&1 && echo "[API RAG] Ya está corriendo en puerto 5001." || \
		(cd $(BASE_DIR) && source $(VENV) && python rag_api_v2.py)

status:
	@echo "=== Estado de servicios ==="
	@pg_lsclusters | grep -q "online" && echo "[PostgreSQL] CORRIENDO" || echo "[PostgreSQL] PARADO"
	@lsof -ti:5000 > /dev/null 2>&1 && echo "[Backend libros :5000] CORRIENDO" || echo "[Backend libros :5000] PARADO"
	@lsof -ti:5001 > /dev/null 2>&1 && echo "[API RAG :5001] CORRIENDO" || echo "[API RAG :5001] PARADO"
	@lsof -ti:3000 > /dev/null 2>&1 && echo "[Frontend :3000] CORRIENDO" || echo "[Frontend :3000] PARADO"
