# Instalar dependencias python
pip install fastapi uvicorn httpx pydantic

# Arrancar el servidor (servidor & index deben estar en la misma carpeta)
uvicorn servidor:app

# Encontrar servidor en localhost:8000
