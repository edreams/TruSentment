import streamlit as st
import os
import requests
import json
from dotenv import load_dotenv
from llama_index import download_loader, SimpleWebPageReader, SimpleDirectoryReader
from io import StringIO
from trulens_eval import Feedback, Tru, feedback as trulens_feedback, LiteLLM
from langchain.llms import VertexAI
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
import time
from google.api_core import exceptions
from llama_index.vector_stores import MilvusVectorStore
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.storage.storage_context import StorageContext
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import as_completed
import pandas as pd
from st_aggrid import AgGrid
from trulens_eval import TruLlama, Feedback, Tru, feedback as trulens_feedback
from trulens_eval.feedback import Groundedness

def setup_page():
    st.set_page_config(
        page_title="TruSentment",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a Bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )
    logo_path='./img/trusent.png'
    st.image(logo_path, use_column_width='auto')

    st.markdown("""
        <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #0a4f87;
            text-align: center;
        }
        button[data-baseweb="button"] {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 16px;
            transition-duration: 0.4s;
        }
        button[data-baseweb="button"]:hover {
            background-color: #45a049;
        }
        </style>
        <div class="title">TruSentment</div>
        """, unsafe_allow_html=True)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['HUGGINGFACE_API_KEY'] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["URI"] = os.getenv("URI")
os.environ["TOKEN"] = os.getenv("TOKEN")
def init_vars():
    if 'tru' not in st.session_state:
        st.session_state['litellm'] = LiteLLM(model_engine="text-bison")
    if 'openai_llm' not in st.session_state:
        st.session_state['openai_llm'] = OpenAI()
    if 'tru' not in st.session_state:
        st.session_state['tru'] =  Tru()
    if 'vertex_llm' not in st.session_state:
        st.session_state['vertex_llm'] = VertexAI(model="text-bison", temperature=0, additional_kwargs={})
    if 'embed_ada' not in st.session_state:
        st.session_state['embed_ada'] = OpenAIEmbeddings()
    
    if 'embed_v12' not in st.session_state:
        st.session_state['embed_v12'] = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if 'embed_vertex' not in st.session_state:
        st.session_state['embed_vertex'] = VertexAIEmbeddings()
    if 'llms' not in st.session_state:
        st.session_state['llms'] = [st.session_state['openai_llm'], st.session_state['vertex_llm']]


dim_ada=1536
dim_v12=384
dim_vertex=768
CHUNK_SIZE = 200  
URI = os.environ.get("URI")
TOKEN = os.environ.get("TOKEN")
TOP_K = 1
INDEX_TYPE = "IVF_FLAT"

def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'json':
            data = json.load(uploaded_file)
            JsonDataReader = download_loader("JsonDataReader")
            loader = JsonDataReader()
            return loader.load_data(data)
        else:
            with open("./tmp/tempfile", "wb") as f:
                f.write(uploaded_file.getbuffer())
            reader = SimpleDirectoryReader("./tmp/", recursive=True)
            return reader.load_data()
    return None

def process_vd(uri, token, dim, collection_name, index_type,metric_type,nprobe): 
    vector_store = MilvusVectorStore(
        uri=uri,
        token=token,
        dim=dim,
        collection_name=collection_name,
        index_params={
            "index_type": index_type,
            "metric_type": metric_type
        },
        search_params={"nprobe": nprobe},
        overwrite=True)
    
    return vector_store

def parte1(secdocuments, embed_model,llm, dim,chunk_size=None):
    vector_store =process_vd(uri=URI,
        token=TOKEN,
        dim=dim,
        collection_name="test2",
        index_type="IVF_FLAT",
        metric_type="L2",
        nprobe=20
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    if chunk_size:
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm, chunk_size=chunk_size)
    else:
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    index = VectorStoreIndex.from_documents(documents=secdocuments, service_context=service_context, storage_context=storage_context)
    
    return index

def parte2():
    openai_gpt35 = trulens_feedback.OpenAI()
    grounded = Groundedness(groundedness_provider=openai_gpt35)

    f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness").on(
        TruLlama.select_source_nodes().node.text.collect()
    ).on_output().aggregate(grounded.grounded_statements_aggregator)

    f_qa_relevance = Feedback(openai_gpt35.relevance_with_cot_reasons, name="Answer Relevance").on_input_output()
    f_qs_relevance = Feedback(openai_gpt35.qs_relevance_with_cot_reasons, name="Context Relevance").on_input().on(
        TruLlama.select_source_nodes().node.text
    ).aggregate(np.max)

    return [f_groundedness, f_qa_relevance, f_qs_relevance]   

def parte4(llms,secdocuments,test_prompts):
          for llm in llms:
              if llm == st.session_state['openai_llm']:
                  DIMENSION=dim_ada
                  embed_model = st.session_state['embed_ada']
              elif llm == st.session_state['vertex_llm']:
                  DIMENSION = dim_v12
                  embed_model = st.session_state['embed_v12']
              vector_store = process_vd(uri=URI,
                                              token=TOKEN,
                                              dim=DIMENSION,
                                              collection_name="test2",
                                              index_type= INDEX_TYPE,
                                              metric_type= "L2",
                                              nprobe= 20)
              
             
              index_comparacion =parte1(secdocuments=secdocuments,embed_model=embed_model,llm=llm,dim=DIMENSION,chunk_size=CHUNK_SIZE )
              query_engine = index_comparacion.as_query_engine(similarity_top_k=TOP_K)
              tru_query_engine = TruLlama(query_engine, feedbacks=parte2(),
                                          metadata={
                                              'index_param': INDEX_TYPE,
                                              'llm_model': llm,
                                              'embed_model': embed_model,
                                              'top_k': TOP_K,
                                              'chunk_size': CHUNK_SIZE
                                          })
              @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
              def call_tru_query_engine(prompt):
                  return query_engine.query(prompt)
              with tru_query_engine as recording:
                  for prompt in test_prompts:
                      try:
                        call_tru_query_engine(prompt)
                      except exceptions.ResourceExhausted:
                          print("Cuota excedida, esperando para reintentar...")
                          time.sleep(60) 
                          call_tru_query_engine(prompt)
                      time.sleep(1)  

def display_results(df):
    AgGrid(df)

    output = StringIO()
    df.info(buf=output)
    info_str = output.getvalue()
    st.text("Información del DataFrame:")
    st.text(info_str)

    
    for index, row in df.iterrows():
        st.text(f"App Id: {row['app_id']}")
        st.text(f"Record Id: {row['record_id']}")
        st.text(f"Input: {row['input']}")
        st.text(f"Output: {row['output']}")
        st.text(f"Answer Relevance: {row['Answer Relevance']}")
        st.text(f"Context Relevance: {row['Context Relevance']}")
        st.text(f"Groundedness: {row['Groundedness']}")
        st.text("-" * 50)  # Línea divisoria entre registros

def obtener_ip_externa():
    try:
        respuesta = requests.get('https://httpbin.org/ip')
        ip = respuesta.json()['origin']
        return ip
    except requests.RequestException:
        return "No se pudo obtener la IP externa"

def asignar_color(metric_openai, metric_vertex):
    if metric_openai > metric_vertex:
        return "🟢", "🔴"  # OpenAI verde, Vertex rojo
    else:
        return "🔴", "🟢"  # OpenAI rojo, Vertex verde



def generate_model_table(openai_cr, openai_g, vertex_cr, vertex_g):
    
    color_cr_openai, color_cr_vertex = asignar_color(openai_cr, vertex_cr)
    color_g_openai, color_g_vertex = asignar_color(openai_g, vertex_g)
    table = "| Modelo | Context Relevance | Groundedness |\n"
    table += "|--------|-------------------|--------------|\n"
    table += f"| OpenAI | {openai_cr} {color_cr_openai} | {openai_g} {color_g_openai} |\n"
    table += f"| VertexAI | {vertex_cr} {color_cr_vertex} | {vertex_g} {color_g_vertex} |\n"
    return table



def main():
    setup_page()
    st.title('TruSentment Analyzer')

    # Option for the user to select the default file or upload their own
    default_file_path = './msft/789019_10K_2023_0000950170-23-035122.json'
    file_selection = st.radio("Choose your file source", ("Use default file MSFT 10-k report", "Upload my file"))

    if file_selection == "Use default file MSFT 10-k report":
        # Directly read the default file
        with open(default_file_path, 'r') as file:
            data = json.load(file)
        secdocuments = data  # Data from the default file
    else:
        uploaded_file = st.file_uploader("Upload a file", type=None)
        if uploaded_file is not None:
            # Process uploaded file
            with open("./tmp/tempfile", "wb") as f:
                f.write(uploaded_file.getbuffer())
            secdocuments = SimpleDirectoryReader("./tmp/", recursive=True).load_data()


    user_input = st.text_area("Write your prompts here, One for line")
    if user_input:
        test_prompts = user_input.split("\n")  

    if user_input and st.button('Execute', key='ejecutar_btn'):
        with st.spinner('Executing... Please wait...'):
            
            init_vars()  

            #Parte 1
            index= parte1(secdocuments=secdocuments,embed_model=st.session_state['embed_v12'], llm=st.session_state['openai_llm'],dim=dim_v12)
            query_engine = index.as_query_engine(top_k=3)

            @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
            def call_query_engine(prompt):
                return query_engine.query(prompt)

            for prompt in test_prompts:
                call_query_engine(prompt)
            #Parte 2
            f_groundedness, f_qa_relevance, f_qs_relevance = parte2()
            
            #Parte 4
            parte4(llms=st.session_state['llms'],secdocuments=secdocuments,test_prompts=test_prompts)
        st.success('Execution Completed!')    
          
            
        #Parte 5
        st.session_state['tru'].run_dashboard()  # Abre una aplicación local de Streamlit para explorar

        #dashboard_url = "http://34.121.120.255:8502"
        dashboard_url = f"http://{obtener_ip_externa()}:8502"
        iframe_html = f"""
            <style>
                .iframe-container {{
                    width: 90%;
                    height: 500px;
                    margin: auto;
                }}
                .iframe-container iframe {{
                    width: 100%;
                    height: 100%;
                }}
            </style>
            <div class="iframe-container">
                <iframe src="{dashboard_url}" allowfullscreen></iframe>
            </div>
        """
        st.markdown(iframe_html, unsafe_allow_html=True)    
        df = st.session_state['tru'].get_records_and_feedback(app_ids=[])[0]
        # Filtrar los DataFrames para OpenAI y VertexAI
        openai_df = df[df['total_cost'] != 0]  # o df[df['total_tokens'] != 0]
        vertex_df = df[df['total_cost'] == 0]  # o df[df['total_tokens'] == 0]


        openai_cr = openai_df['Context Relevance'].mean()
        openai_g = openai_df['Groundedness'].mean()
        vertex_cr = vertex_df['Context Relevance'].mean()
        vertex_g = vertex_df['Groundedness'].mean()

        table_md = generate_model_table(openai_cr, openai_g, vertex_cr, vertex_g)
        st.markdown(table_md)
        display_results(df)
        st.write(st.session_state['tru'].get_records_and_feedback(app_ids=[])[0])  # Pasar una lista vacía de app_ids para obtener todo

if __name__ == '__main__':
    main()