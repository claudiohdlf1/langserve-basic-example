# Importación de bibliotecas necesarias para el proyecto

# Biblioteca estándar de Python para interactuar con el sistema operativo
import os

# Componentes de LangChain Community para cargar y procesar documentos
from langchain_community.document_loaders import TextLoader  # Para cargar archivos de texto
from langchain_community.vectorstores import FAISS  # Base de datos vectorial para búsqueda semántica

# Componentes de LangChain OpenAI para embeddings y chat
from langchain_openai import OpenAIEmbeddings  # Para convertir texto en vectores
from langchain_openai import ChatOpenAI  # Para interactuar con modelos de chat de OpenAI

# Utilidades de LangChain para procesamiento de texto
from langchain_text_splitters import CharacterTextSplitter  # Para dividir textos en chunks

# Componentes core de LangChain para manejo de prompts y flujos
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Para crear templates de chat
from langchain_core.output_parsers import StrOutputParser  # Para parsear respuestas como strings
from langchain_core.runnables import RunnablePassthrough  # Para crear pipelines de procesamiento

# Utilidades para manejo de variables de entorno
import dotenv  # Para cargar variables de entorno desde archivo .env

# Componentes adicionales de LangChain para procesamiento avanzado
from langchain.output_parsers.openai_tools import PydanticToolsParser  # Parser específico para herramientas OpenAI
from langchain.schema.runnable import RunnableLambda  # Para crear funciones ejecutables en pipelines

# Pydantic para validación de datos y creación de modelos
from pydantic import BaseModel, Field  # Para definir estructuras de datos con validación

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings


# Biblioteca estándar para manejo de tiempo
import time  # Para mediciones de tiempo y delays

# Configuración inicial del entorno y variables clave

dotenv.load_dotenv()
# Obtenemos la API key desde las variables de entorno
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
# Definimos la ruta local del archivo de datos
DATA_PATH = 'app/state_of_the_union.txt'

# Parámetros de configuración
# Definimos el modelo de OpenAI que vamos a utilizar
MODEL_NAME = "gpt-4o-mini"  # Modelo GPT-4 optimizado para nuestro caso de uso

# Función para obtener los documentos
def get_documents(data_path):
    """
    Carga y divide documentos desde una ruta de archivo especificada.

    Esta función implementa la carga y división de documentos para su posterior
    procesamiento en el sistema RAG.

    Parameters
    ----------
    data_path : str
        Ruta al archivo que contiene el documento a procesar.

    Returns
    -------
    list
        Lista de fragmentos de texto divididos del documento original.

    Notes
    -----
    La función utiliza TextLoader para cargar el documento y CharacterTextSplitter
    para dividirlo en fragmentos manejables.
    """

    # Cargamos el documento usando TextLoader
    # TextLoader es una utilidad de LangChain para cargar archivos de texto
    loader = TextLoader(data_path)
    documents = loader.load()

    # Creamos el divisor de texto y procesamos el documento
    # Configuramos un tamaño de chunk de 1000 caracteres sin superposición
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Dividimos los documentos en fragmentos más pequeños
    texts = text_splitter.split_documents(documents)
    # Llamamos a la función para guardar los documentos en qdrant
    save_documents_to_qdrant(texts)

    return texts

# Función que guarda los documentos en qdrant
def save_documents_to_qdrant(texts):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY
        )

    qdrant = QdrantVectorStore.from_documents(
        texts,
        embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True,
        collection_name="state_of_the_union_3",
        force_recreate=True
    )



# Función para obtener el recuperador (retriever)
def get_retriever(texts, k=20):
    """
    Crea un objeto recuperador (retriever) a partir de los textos proporcionados.

    Esta función genera embeddings para los textos usando la API de OpenAI,
    crea un almacén de vectores usando FAISS y devuelve un objeto recuperador
    configurado con los parámetros de búsqueda especificados.

    Parameters
    ----------
    texts : list
        Lista de documentos de texto que serán convertidos a embeddings y almacenados.
    k : int, optional
        Número de vecinos más cercanos a recuperar (default es 20).

    Returns
    -------
    VectorStoreRetriever
        Objeto recuperador para buscar en el almacén de vectores.

    Notes
    -----
    El recuperador utiliza FAISS como motor de búsqueda de similitud vectorial
    y OpenAI para generar los embeddings de los textos.
    """
    # Generamos los embeddings para los textos usando la API de OpenAI
    # Los embeddings son representaciones numéricas del significado del texto
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Creamos el almacén de vectores a partir de los documentos y embeddings
    # FAISS es una biblioteca eficiente para búsqueda de similitud
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Generamos y retornamos el objeto recuperador con los parámetros de búsqueda
    # El parámetro k determina cuántos documentos similares se recuperarán
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever

# Implementación de ReRank para mejorar la relevancia de los resultados
class RelevanceScore(BaseModel):
    """Modelo para almacenar el puntaje de relevancia."""
    score: float = Field(..., description="Puntaje de relevancia entre 0 y 1")

def get_relevance_score(query, docs, model="gpt-4o"):
    """
    Evalúa y clasifica los documentos según su relevancia con respecto a la consulta del usuario.

    Parameters
    ----------
    query : str
        La consulta o pregunta del usuario.
    docs : list
        Lista de documentos a evaluar y clasificar.

    Returns
    -------
    list
        Lista de puntajes de relevancia para los documentos proporcionados.

    Notes
    -----
    Esta función utiliza un modelo de lenguaje para analizar el contenido de cada documento
    y su relación con la consulta, asignando un puntaje de relevancia entre 0 y 1
    (siendo 1 el más relevante).
    """
    # Inicializamos el modelo de lenguaje con los parámetros especificados
    # Usamos temperatura baja (0.1) para obtener respuestas más consistentes
    model = ChatOpenAI(model=model, temperature=0.1, api_key=OPENAI_API_KEY)

    # Vinculamos la herramienta RelevanceScore al modelo
    # Esto permite que el modelo genere puntajes de relevancia estructurados
    model_with_tools = model.bind_tools([RelevanceScore], tool_choice=RelevanceScore.__name__)

    # Configuramos el parser para procesar las respuestas del modelo
    # y convertirlas en objetos RelevanceScore
    parser = PydanticToolsParser(tools=[RelevanceScore])

    # Definimos la plantilla del prompt que guiará al modelo
    # en la tarea de evaluación y clasificación
    prompt = ChatPromptTemplate.from_template("""Eres un experto en evaluar y clasificar resultados de búsqueda. Tu tarea es reclasificar los documentos dados según su relevancia con la consulta del usuario.

    # Instrucciones:
    - Analiza cuidadosamente el contenido de cada documento y su relación con la consulta
    - Asigna un puntaje de relevancia entre 0 y 1 para cada documento (1 siendo el más relevante)

    Consulta del usuario: {query}

    Documento a clasificar:
    {document}
    """)

    # Creamos la cadena de procesamiento combinando el prompt, modelo y parser
    chain_single = prompt | model_with_tools | parser

    # Preparamos el contexto para procesamiento paralelo
    # Transformamos los documentos en el formato requerido por la cadena
    prep_context = RunnableLambda(lambda x: [{"query": query, "document": doc.page_content} for doc in docs])

    # Combinamos la preparación del contexto con la cadena de procesamiento
    chain = prep_context | chain_single.map()

    # Ejecutamos la cadena para obtener los puntajes de relevancia
    response = chain.invoke(docs)

    return response

def filter_docs(relevance_scores, docs, threshold=0.5):
    """
    Filtra los documentos basándose en sus puntajes de relevancia.

    Esta función filtra los documentos que tienen puntajes de relevancia por debajo
    del umbral especificado, ordena los documentos restantes por puntaje de relevancia
    en orden descendente y retorna la lista filtrada.

    Parameters
    ----------
    relevance_scores : list
        Lista de puntajes de relevancia para los documentos dados.
    docs : list
        Lista de documentos a ser filtrados y ordenados.
    threshold : float, optional
        Puntaje mínimo de relevancia para mantener un documento, por defecto 0.5.

    Returns
    -------
    list
        Lista de documentos filtrados y ordenados por relevancia.
    """
    # Asignamos los puntajes de relevancia a los metadatos de cada documento
    # Esto nos permite mantener el puntaje asociado a cada documento
    for doc, res in zip(docs, relevance_scores):
        doc.metadata['relevance_score'] = res[0].score

    # Ordenamos los documentos por puntaje de relevancia de mayor a menor
    # Utilizamos una función lambda para acceder al puntaje en los metadatos
    docs.sort(key=lambda doc: doc.metadata['relevance_score'], reverse=True)

    # Filtramos los documentos, manteniendo solo aquellos con puntaje >= threshold
    # Usamos list comprehension para crear una nueva lista con los documentos filtrados
    docs = [doc for doc in docs if doc.metadata['relevance_score'] >= threshold]

    return docs

def generate_response(question, docs, model='gpt-4o'):
    """
    Genera una respuesta para una pregunta utilizando el contexto recuperado.

    Esta función crea un prompt para una tarea de preguntas y respuestas, invoca la cadena RAG
    para obtener la respuesta y utiliza el modelo de lenguaje especificado.

    Parameters
    ----------
    question : str
        La pregunta que debe ser respondida
    docs : list
        Lista de documentos que contienen el contexto para responder la pregunta
    model : str, optional
        Nombre del modelo de lenguaje a utilizar (default es 'gpt-4o')

    Returns
    -------
    str
        La respuesta generada por el modelo
    """

    # Creamos la plantilla del prompt que define cómo el modelo debe responder
    # Incluimos instrucciones específicas para responder en español y de forma concisa
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    # Instructions
    - Answer in spanish

    Question: {question}

    Context: {context}

    Answer:""")

    # Inicializamos el modelo de lenguaje con los parámetros especificados
    # Usamos temperature=0.1 para obtener respuestas más deterministas
    model = ChatOpenAI(model=model, temperature=0.1, api_key=OPENAI_API_KEY)

    # Creamos la cadena RAG combinando el prompt, el modelo y el parser
    # Utilizamos el operador pipe (|) para encadenar los componentes
    rag_chain = prompt | model | StrOutputParser()

    # Unimos el contenido de todos los documentos en un solo texto de contexto
    # Utilizamos list comprehension para extraer el contenido de cada documento
    context = " ".join([doc.page_content for doc in docs])

    # Generamos la respuesta invocando la cadena RAG con la pregunta y el contexto
    response = rag_chain.invoke({'question': question, 'context': context})

    return response

def main(pregunta):
    # Definimos la pregunta que queremos realizar al sistema
    #question = "What did the president say about ketanji brown jackson?"
    question = pregunta
    # Cargamos los documentos desde la ruta especificada
    texts = get_documents(DATA_PATH)

    # Creamos el recuperador (retriever) que nos permitirá buscar documentos relevantes
    retriever = get_retriever(texts)

    # Recuperamos los documentos relacionados con la pregunta
    docs = retriever.invoke(question)

    # Calculamos el puntaje de relevancia para cada documento
    relevance = get_relevance_score(question, docs, model="gpt-4o-mini")

    # Filtramos los documentos usando un umbral de relevancia de 0.2
    filtered_docs = filter_docs(relevance, docs, threshold=0.2)

    # Generamos la respuesta final utilizando los documentos filtrados
    response = generate_response(question, filtered_docs, model="gpt-4o")

    # Mostramos la respuesta
    #print(response)
    return response

#if __name__ == "__main__":
#    question = "What did the president say about ketanji brown jackson?"
#    print(main(question))