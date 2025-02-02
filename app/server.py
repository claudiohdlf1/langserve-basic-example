from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from app.rag import main
import uvicorn


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="RAG App",
)

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Question Answering</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; }
        textarea, input { width: 80%; font-size: 16px; padding: 10px; }
        #response { height: 150px; }
        #loading { display: none; font-weight: bold; color: blue; }
    </style>
</head>
<body>
    <h2>Ingrese su pregunta:</h2>
    <textarea id="question" placeholder="Escriba su pregunta aquí" rows="4"></textarea>
    <br>
    <button onclick="getAnswer()">Enviar</button>
    <p id="loading">Buscando respuesta...</p>
    <h2>Respuesta:</h2>
    <textarea id="response" rows="6" readonly></textarea>
    
    <script>
        async function getAnswer() {
            let question = document.getElementById("question").value;
            let responseBox = document.getElementById("response");
            let loadingText = document.getElementById("loading");
            
            responseBox.value = "";
            loadingText.style.display = "block";
            
            let response = await fetch(`/get_answer?question=${encodeURIComponent(question)}`);
            let data = await response.json();
            
            responseBox.value = data.response;
            loadingText.style.display = "none";
        }
    </script>
</body>
</html>
"""

@app.get("/app", response_class=HTMLResponse)
def serve_page():
    return html_content

@app.get("/get_answer")
def get_answer(question: str = Query(..., title="User Question")):
    """Obtiene una respuesta desde la función main() de rag.py basada en la pregunta del usuario."""
    return {"response": main(question)}

if __name__ == "__main__":
    import uvicorn

    #uvicorn.run(app, host="localhost", port=8080)
    uvicorn.run("app.server:app", host="0.0.0.0", port=8080)
    #uvicorn.run(app, host="localhost", port=8080, reload=True)