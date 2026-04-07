import streamlit as st
import requests
from google import genai
import os
from bs4 import BeautifulSoup
import base64
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURAÇÕES ---


SYSTEM_INSTRUCTION = """
Você é o Rodrigo GPT, um agente técnico da E-Deploy.
Se perguntarem quem é você: "Sou Rodrigo GPT, fã de churros."
"""

# --- CLIENTE GEMINI ---
try:
    client = genai.Client(api_key=api_key)
    print("✅ Cliente Gemini OK")
except Exception as e:
    st.error(f"Erro ao iniciar cliente: {e}")
    st.stop()

# --- LIMPEZA HTML ---
def limpar_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(["nav", "header", "footer", "style", "script"]):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)

# --- BUSCAR CONFLUENCE ---
def buscar_confluence(space_key):
    auth = (USER_EMAIL.strip(), API_TOKEN.strip())
    url = f"{CONFLUENCE_URL}/wiki/rest/api/space/{space_key}/content/page?expand=body.storage&limit=50"

    try:
        res = requests.get(url, auth=auth)

        if res.status_code == 404:
            url = f"{CONFLUENCE_URL}/rest/api/space/{space_key}/content/page?expand=body.storage&limit=50"
            res = requests.get(url, auth=auth)

        res.raise_for_status()
        data = res.json()

        results = data.get("results", [])
        return [
            {
                "title": p.get("title"),
                "text": limpar_html(p.get("body", {}).get("storage", {}).get("value", ""))
            }
            for p in results if p.get("body")
        ]

    except Exception as e:
        print("Erro Confluence:", e)
        return []

# --- VECTOR STORE ---
def create_vector_store(kb):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)

    docs = []
    for item in kb:
        chunks = splitter.split_text(item["text"])
        for c in chunks:
            docs.append({"title": item["title"], "text": c})

    texts = [d["text"] for d in docs]
    embeddings = []

    st.sidebar.info("Gerando embeddings...")
    bar = st.sidebar.progress(0)

    try:
        for i in range(0, len(texts), 20):
            batch = texts[i:i+20]

            res = client.models.embed_content(
                model="embedding-001",  # ✅ CORRIGIDO
                contents=batch
            )

            for emb in res.embeddings:
                embeddings.append(emb.embedding)  # ✅ CORRIGIDO

            bar.progress(min((i+20)/len(texts), 1.0))

        if not embeddings:
            raise ValueError("Nenhum embedding gerado.")

        emb_array = np.array(embeddings).astype("float32")

        index = faiss.IndexFlatL2(emb_array.shape[1])
        index.add(emb_array)

        return index, docs

    except Exception as e:
        st.error(f"Erro no embedding: {e}")
        return None, None

# --- JIRA ---
def buscar_jira(query):
    try:
        url = f"{CONFLUENCE_URL}/rest/api/3/search/jql"

        auth = base64.b64encode(f"{USER_EMAIL}:{API_JIRA}".encode()).decode()

        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/json"
        }

        payload = {
            "jql": f'text ~ "{query}" ORDER BY updated DESC',
            "maxResults": 3
        }

        res = requests.post(url, headers=headers, json=payload)
        issues = res.json().get("issues", [])

        return [f"{i['key']} - {i['fields']['summary']}" for i in issues]

    except:
        return []

# --- RAG ---
def gerar_resposta(query, index, docs, file):

    if index is None:
        return "Erro: base vetorial não carregada."

    try:
        res = client.models.embed_content(
            model="embedding-001",
            contents=[query]
        )

        q_vec = np.array(res.embeddings[0].embedding).astype("float32").reshape(1, -1)

        D, I = index.search(q_vec, k=5)

        context = "\n---\n".join([docs[i]["text"] for i in I[0] if i != -1])

        jira = buscar_jira(query)

        contents = []

        if file:
            contents.append({
                "inline_data": {
                    "data": base64.b64encode(file.read()).decode(),
                    "mime_type": file.type
                }
            })

        prompt = f"""
{SYSTEM_INSTRUCTION}

Pergunta:
{query}

JIRA:
{jira}

Confluence:
{context}
"""

        contents.append(prompt)

        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents
        )

        return resp.text

    except Exception as e:
        return f"Erro: {e}"

# --- INTERFACE ---
st.set_page_config(page_title="Rodrigo GPT", layout="wide")
st.title("🤖 Rodrigo GPT - Suporte Técnico")

uploaded_file = st.file_uploader("Enviar imagem", type=["png", "jpg", "jpeg"])

# --- LOAD BASE ---
if "index" not in st.session_state:
    with st.spinner("Carregando base..."):
        kb = []
        for s in SPACE_KEYS:
            kb.extend(buscar_confluence(s))

        if not kb:
            st.error("Erro ao carregar Confluence")
            st.stop()

        idx, docs = create_vector_store(kb)

        if idx is None:
            st.stop()

        st.session_state.index = idx
        st.session_state.docs = docs
        st.success("Base carregada!")

# --- CHAT ---
if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if pergunta := st.chat_input("Digite sua dúvida"):
    st.session_state.chat.append({"role": "user", "content": pergunta})

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.spinner("Pensando..."):
        resposta = gerar_resposta(
            pergunta,
            st.session_state.index,
            st.session_state.docs,
            uploaded_file
        )

    with st.chat_message("assistant"):
        st.markdown(resposta)

    st.session_state.chat.append({"role": "assistant", "content": resposta})