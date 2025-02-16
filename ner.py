from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
import streamlit as st
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de variables de entorno
os.environ['LANCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "pr-IntroductionLangChain"

# Cargar múltiples PDFs
def cargar_pdfs(pdf_paths):
    documentos = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        documentos.extend(loader.load())  # Agrega el contenido del PDF a la lista
    return documentos

# Cargar los 3 PDFs
pdf_paths = ["Datos_suplementos.pdf", "Movimiento_musculacion.pdf"]
documentos = cargar_pdfs(pdf_paths)

# Verificar si los documentos se cargaron correctamente
if not documentos:
    st.error("No se cargaron documentos. Verifique la ubicación y el formato de los PDFs.")
else:
    # Usar una variable de estado para indicar que los documentos se cargaron correctamente
    st.session_state.docs_loaded = True

# Generar embeddings y crear un retriever con FAISS
try:
    embeddings = OllamaEmbeddings(model="llama3.2")  # Embedding de Ollama
    faiss = FAISS.from_documents(documentos, embeddings)
    retriever = faiss.as_retriever()  # Usar FAISS como el retriever
    if 'docs_loaded' in st.session_state:  # Verifica si los documentos fueron cargados
        # Usar una variable de estado para indicar que FAISS se creó correctamente
        st.session_state.faiss_created = True
except Exception as e:
    st.error(f"Error al crear los embeddings o FAISS: {str(e)}")

# Configuración del modelo y el prompt
llm = OllamaLLM(model="llama3.2")
output_parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "Eres un entrenador virtual experto en fitness y nutrición. Ayudas a las personas a alcanzar sus metas físicas ofreciendo rutinas de ejercicio personalizadas, "
         "corrigiendo técnicas de ejercicios y dando consejos alimentarios basados en sus objetivos. Explicas conceptos de forma clara y adaptas tus respuestas a principiantes o avanzados. "
         "También das recomendaciones sobre suplementos, destacando beneficios y posibles riesgos. Mantén un tono motivador y profesional. Tus respuestas deben ser detalladas, pero fáciles de entender. "
         "Además de lo anterior, proporcionas apoyo continuo y motivación, alentando a tus clientes a superar sus límites y mantener la disciplina. Ayudas a establecer metas realistas y alcanzables, "
         "realizando un seguimiento regular del progreso y ajustando los planes según sea necesario. "
         "Funciones adicionales: Asesoría Personalizada: Realizas evaluaciones iniciales para entender el nivel de fitness, objetivos y posibles limitaciones físicas de cada persona. "
         "Planificación de Dietas: Diseñas planes de alimentación detallados y balanceados, teniendo en cuenta las preferencias alimentarias, alergias y objetivos nutricionales. "
         "Guía de Recuperación: Proporcionas estrategias y ejercicios de recuperación para prevenir lesiones y asegurar una recuperación adecuada después de los entrenamientos intensos. "
         "Educación Continua: Ofreces información sobre los últimos estudios y tendencias en fitness y nutrición para mantener a tus clientes informados y motivados. "
         "Soporte Emocional: Actúas como un apoyo emocional para ayudar a tus clientes a superar barreras mentales y mantener una actitud positiva. "
         "Entrenamiento a Distancia: Utilizas herramientas tecnológicas para ofrecer entrenamientos virtuales en tiempo real, así como seguimiento y soporte continuo a través de aplicaciones y plataformas de comunicación. "
         "Recomendaciones de Equipamiento: Aconsejas sobre el mejor equipamiento y herramientas para entrenar en casa o en el gimnasio, adaptadas a las necesidades individuales. "
         "En todo momento, mantienes una actitud positiva y motivadora, incentivando a tus clientes a alcanzar sus objetivos de forma segura y eficiente."
        ),
        ("user", "Question: {question}")
    ]
)

# Cadena de recuperación
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# Personalización de la interfaz con CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #008080;
            font-size: 36px;
            text-align: center;
        }
        .stButton>button {
            background-color: #ff6347;
            color: white;
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True
)

# Título y descripción
st.markdown('<h1 class="title">Bienvenido al Asistente de Fitness Virtual</h1>', unsafe_allow_html=True)
st.write("Soy tu entrenador virtual, listo para ayudarte con rutinas, nutrición y más.")

# Verificar si ya existe el historial de mensajes
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message[0] == 'user':  # Accede al primer elemento de la tupla
        with st.chat_message("user"):
            st.markdown(message[1])  # Accede al segundo elemento de la tupla
    else:
        with st.chat_message("assistant"):
            st.markdown(message[1])  # Accede al segundo elemento de la tupla


# Función del chatbot
def chatbot(input_text, messages, topic):
    # Agregar el mensaje del usuario al historial (en formato adecuado)
    messages.append(('user', input_text))
    
    # Crear el diccionario de entrada que debe incluir 'chat_history' como una lista de tuplas
    inputs = {
        'question': input_text,
        'chat_history': messages  # Historial en formato de tuplas
    }
    
    # Llamamos al 'chain' con el historial de chat incluido
    response = chain(inputs)
    
    # Agregar la respuesta del asistente al historial (en formato adecuado)
    messages.append(('assistant', response['answer']))
    
    return response['answer'], messages

# Entrada del usuario
user_input = st.text_input("Hazme cualquier pregunta sobre fitness:")
topic = st.selectbox('Selecciona un tema:', ['Suplementos', 'Rutinas de ejercicio', 'Nutrición'])

if user_input:
    response, st.session_state.messages = chatbot(user_input, st.session_state.messages, topic)
    st.markdown(f"**Respuesta:** {response}")

# Botón para reiniciar la conversación
if st.button("Reiniciar conversación"):
    st.session_state.messages = []
    st.experimental_rerun()