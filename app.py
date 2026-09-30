import os
import sqlite3
import pandas as pd
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv, find_dotenv
from bs4 import BeautifulSoup
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from openai import AzureOpenAI

# Carrega e força o mapeamento do arquivo .env local
load_dotenv(find_dotenv(), override=True)

app = Flask(__name__)

# --- CONFIGURAÇÃO E VALIDAÇÃO DAS VARIÁVEIS ---
ENDPOINT = os.getenv("ENDPOINT")
API_KEY = os.getenv("API_KEY")
API_VERSION = os.getenv("API_VERSION", "2024-08-01-preview")
MODELO = os.getenv("MODELO")

print("--- DIAGNÓSTICO DE CREDENCIAIS ---")
print(f"ENDPOINT: {ENDPOINT}")
print(f"API_KEY: {'Definida (' + str(len(API_KEY)) + ' caracteres)' if API_KEY else 'NÃO ENCONTRADA'}")
print(f"API_VERSION: {API_VERSION}")
print(f"MODELO: {MODELO}")
print("-----------------------------------")

if not all([ENDPOINT, API_KEY, MODELO]):
    print("⚠️ ERRO CRÍTICO: ENDPOINT, API_KEY e MODELO devem estar configurados no arquivo .env.")

azure_client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION
)

# --- CONFIGURAÇÃO DO BANCO DE DADOS SQLITE3 ---
DB_NAME = "historico_rag.db"

def init_sqlite():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pergunta TEXT NOT NULL,
            resposta TEXT NOT NULL,
            contexto_usado TEXT,
            data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_sqlite()

# --- MODELO DE EMBEDDINGS E CHROMA DB ---
print("Carregando modelo de embeddings local ('all-MiniLM-L6-v2')...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.Client()
collection_name = "base_educacional"
try:
    chroma_client.delete_collection(collection_name)
except Exception:
    pass

vector_collection = chroma_client.create_collection(name=collection_name)

def fatiar_texto(texto, tamanho=500, overlap=50):
    """Função de fatiamento de texto com tamanho fixo e sobreposição."""
    if not texto:
        return []
    chunks = []
    inicio = 0
    comprimento = len(texto)
    while inicio < comprimento:
        fim = min(inicio + tamanho, comprimento)
        chunk = texto[inicio:fim].strip()
        if chunk:
            chunks.append(chunk)
        inicio += (tamanho - overlap)
    return chunks

def ingestar_dados():
    """Realiza a ingestão, limpeza, fatiamento e indexação das 3 fontes de dados."""
    todos_chunks = []
    todos_metadados = []
    todos_ids = []
    contador_id = 0


    # 2. Arquivo Tabular (dados.csv)
    if os.path.exists("dados.csv"):
        try:
            df = pd.read_csv("dados.csv")
            for _, row in df.iterrows():
                texto_linha = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                chunks_csv = fatiar_texto(texto_linha)
                for c in chunks_csv:
                    todos_chunks.append(c)
                    todos_metadados.append({"fonte": "Arquivo Tabular (dados.csv)"})
                    todos_ids.append(f"csv_{contador_id}")
                    contador_id += 1
        except Exception as e:
            print(f"Erro ao ler dados.csv: {e}")

    # 3. Documento PDF (escola.pdf)
    if os.path.exists("escola.pdf"):
        try:
            reader = PdfReader("escola.pdf")
            texto_pdf = ""
            for pagina in reader.pages:
                txt = pagina.extract_text()
                if txt:
                    texto_pdf += txt + "\n"
            chunks_pdf = fatiar_texto(texto_pdf)
            for c in chunks_pdf:
                todos_chunks.append(c)
                todos_metadados.append({"fonte": "Documento PDF (escola.pdf)"})
                todos_ids.append(f"pdf_{contador_id}")
                contador_id += 1
        except Exception as e:
            print(f"Erro ao ler escola.pdf: {e}")



    # 1. Web Scraping
    url = "https://gratuitos.netlify.app/"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            texto_web = soup.get_text(separator=' ')
            chunks_web = fatiar_texto(texto_web)
            for c in chunks_web:
                todos_chunks.append(c)
                todos_metadados.append({"fonte": "Web Scraping (gratuitos.netlify.app)"})
                todos_ids.append(f"web_{contador_id}")
                contador_id += 1
    except Exception as e:
        print(f"Erro ao fazer scraping da URL: {e}")            

    # Inserção no ChromaDB
    if todos_chunks:
        embeddings = embedding_model.encode(todos_chunks).tolist()
        vector_collection.add(
            documents=todos_chunks,
            embeddings=embeddings,
            metadatas=todos_metadados,
            ids=todos_ids
        )
        print(f"✅ Ingestão concluída com sucesso! Total de chunks indexados: {len(todos_chunks)}")
    else:
        print("⚠️ Nenhum dado encontrado para indexação.")

# Executa a ingestão ao iniciar a aplicação
ingestar_dados()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/perguntar", methods=["POST"])
def perguntar():
    data = request.get_json()
    pergunta = data.get("pergunta", "").strip()

    if not pergunta:
        return jsonify({"erro": "A pergunta não pode estar vazia."}), 400

    # Recuperação Vetorial (Top-K = 3)
    vetor_pergunta = embedding_model.encode([pergunta]).tolist()
    resultados = vector_collection.query(
        query_embeddings=vetor_pergunta,
        n_results=3
    )

    trechos_recuperados = resultados.get("documents", [[]])[0]
    metadados_recuperados = resultados.get("metadatas", [[]])[0]

    # Constrói o contexto consolidado
    contexto_consolidado = "\n\n---\n\n".join(trechos_recuperados) if trechos_recuperados else "Nenhum contexto encontrado."

    # Prompt de sistema restrito para RAG
    system_prompt = (
        "Você é um assistente educacional especialista. Responda à pergunta do usuário "
        "utilizando estritamente e apenas as informações contidas no contexto fornecido abaixo. "
        "Se a resposta não puder ser encontrada no contexto, informe educadamente que não possui "
        "informações suficientes nas fontes institucionais."
    )

    user_prompt = f"Contexto:\n{contexto_consolidado}\n\nPergunta: {pergunta}"

    try:
        response = azure_client.chat.completions.create(
            model=MODELO,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
           
            
        )
        resposta_llm = response.choices[0].message.content
    except Exception as e:
        resposta_llm = f"Erro ao comunicar com a API da Azure OpenAI: {str(e)}"

    # Salva interação no SQLite3
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO interacoes (pergunta, resposta, contexto_usado) VALUES (?, ?, ?)",
            (pergunta, resposta_llm, contexto_consolidado)
        )
        conn.commit()
        conn.close()
    except Exception as db_err:
        print(f"Erro ao salvar no SQLite: {db_err}")

    # Prepara auditoria para o front-end
    auditoria = []
    for doc, meta in zip(trechos_recuperados, metadados_recuperados):
        auditoria.append({"fonte": meta.get("fonte", "Desconhecida"), "texto": doc})

    return jsonify({
        "resposta": resposta_llm,
        "auditoria": auditoria
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)