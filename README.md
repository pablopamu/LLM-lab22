# Instalar dependencias python // Install python dependencies
pip install fastapi uvicorn httpx pydantic

# Arrancar el servidor // Run server
(servidor & index deben estar en la misma carpeta)
(servidor & index in the same directory)
uvicorn servidor:app
localhost:8000

# Uso
Create the two files in the same directoy, activate the enviroment, install the dependencies, run 'uvicorn servidor:app' and open localhost:8000 in a browser.
