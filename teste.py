import streamlit as st
import requests
import google.generativeai as sdk_classico  # ESSENCIAL: Resolve o erro 404/403 de embedding
from google import genai 
import os
from bs4 import BeautifulSoup
import base64
import json
import numpy as np 
import faiss 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import re 

# --- 1. CONFIGURAÇÕES E VARIÁVEIS ---


SYSTEM_INSTRUCTION = """
Você é o **Rodrigo GPT**, um Agente de Suporte Técnico da E-DEPLOY. Sua função é ser proativo, respeitoso e fornecer soluções e procedimentos claros, baseados estritamente nos contextos fornecidos (JIRA e Confluence).

**REGRAS DE CONDUTA:**
1. **Persona:** Se a pergunta for "quem é voce?", responda: "Olá, sou Rodrigo GPT, um grande fã de churros e comida."
2. **Prioridade na Solução:** A resposta deve ser extraída dos contextos. Não use links externos, EXCETO o link do ticket JIRA específico encontrado (ex: 'Link: https://.../browse/TICKET-123').

**PRIORIDADE DE CONTEXTO E GERAÇÃO:**
1. **Confluence RAG (Primeira Prioridade):** Se o JIRA não fornecer uma solução, utilize **APENAS** o 'CONTEXTO DE PROCEDIMENTO' (Confluence) para gerar um passo a passo.
2. **JIRA (Segunda Prioridade e Foco):** O bloco 'CONTEXTO JIRA PRIORITÁRIO' é sua fonte principal. Seu objetivo é identificar a similaridade.
3. **Análise Multimodal:** Se uma imagem ou log for anexado, use a análise dedutiva antes de aplicar o RAG.

**LIMITE DE CONHECIMENTO E FALLBACK:**
"Não encontrei o procedimento solicitado na Base de Conhecimento. Peço que solicite ajuda interna para lhe ajudar com isso. Vou voltar a comer meu Churros."
"""

# Inicialização Híbrida (Novo para Chat / Clássico para Embedding)
try:
    # Cliente para Chat (SDK Novo)
    client = genai.Client(api_key=api_key)
    
    # FORÇANDO A VERSÃO V1 NO SDK CLÁSSICO PARA MATAR O ERRO 404
    from google.generativeai import client as sdk_client
    sdk_classico.configure(api_key=api_key, transport='rest') # Usando REST para maior compatibilidade
    
    print("--- CLIENTES GEMINI INICIALIZADOS (V1 FORCED) ---")
except Exception as e:
    st.error(f"Erro ao iniciar o cliente: {e}")
    st.stop()

# --- 2. FUNÇÕES DE SUPORTE ---

def limpando_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(["nav", "header", "footer", "style", "script"]):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)

def busca_conteudo_confluence(space_key):
    auth_credentials = (USER_EMAIL.strip(), API_TOKEN.strip()) 
    headers = {"Accept": "application/json"}
    base_url = CONFLUENCE_URL.rstrip('/')
    url = f"{base_url}/wiki/rest/api/space/{space_key}/content/page?expand=body.storage&limit=50"

    try:
        response = requests.get(url, headers=headers, auth=auth_credentials)
        if response.status_code == 404:
            url = f"{base_url}/rest/api/space/{space_key}/content/page?expand=body.storage&limit=50"
            response = requests.get(url, headers=headers, auth=auth_credentials)
        
        response.raise_for_status()
        data = response.json()
        results = data.get('page', {}).get('results', []) if 'page' in data else data.get('results', [])
        
        pages = []
        for page in results:
            title = page.get('title')
            content = page.get('body', {}).get('storage', {}).get('value', '')
            if content:
                pages.append({"title": title, "text": limpando_html_content(content)})
        return pages
    except Exception as e:
        print(f"Erro no Confluence ({space_key}): {e}")
        return []

def create_vector_store(knowledge_base):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    documents = []
    for item in knowledge_base:
        chunks = text_splitter.split_text(item['text'])
        for chunk in chunks:
            documents.append({"title": item['title'], "text": chunk})
    
    texts = [doc["text"] for doc in documents]
    all_embeddings = []
    
    # Progresso na barra lateral como no original
    st.sidebar.info(f"Gerando embeddings para {len(texts)} chunks...")
    progress_bar = st.sidebar.progress(0.0)
    
    try:
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            
            # CHAMADA REVISADA: Forçando a API Version v1 aqui
            res = sdk_classico.embed_content(
                model='models/embedding-001',
                content=batch,
                task_type="retrieval_document",
                request_options={'api_version': 'v1'} # FORÇA A V1 AQUI
            )
            all_embeddings.extend(res['embedding'])
            progress_bar.progress((i + len(batch)) / len(texts))
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, documents
    except Exception as e:
        st.error(f"Erro fatal no Embedding: {e}")
        return None, None

def busca_chamados_jira(user_query, max_results=3):
    jira_url = f"{CONFLUENCE_URL}/rest/api/3/search/jql"
    auth_string = f"{USER_EMAIL}:{API_JIRA}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    headers = {"Authorization": f"Basic {encoded_auth}", "Content-Type": "application/json", "Accept": "application/json"}
    
    jql = f'text ~ "{user_query}" ORDER BY updated DESC'
    payload = {"jql": jql, "fields": ["key", "summary", "status", "description", "comment"], "maxResults": max_results}
    
    try:
        response = requests.post(jira_url, headers=headers, json=payload)
        response.raise_for_status()
        issues = response.json().get('issues', [])
        tickets = []
        for issue in issues:
            key = issue['key']
            summary = issue['fields']['summary']
            tickets.append(f"Ticket: {key} - {summary}\nLink: {CONFLUENCE_URL}/browse/{key}")
        return tickets
    except:
        return []

def gerar_resposta_rag(user_query, vector_index, documents, uploaded_file):
    # 1. Busca JIRA
    jira_tickets = busca_chamados_jira(user_query)
    jira_context = "\n".join(jira_tickets) if jira_tickets else "Nenhum ticket JIRA encontrado."

    # 2. Embedding da Query (SDK CLÁSSICO)
    # No início da função gerar_resposta_rag:
    res_query = sdk_classico.embed_content(
        model='models/embedding-001',
        content=user_query,
        task_type="retrieval_query",
        request_options={'api_version': 'v1'} # FORÇA A V1 AQUI TAMBÉM
    )
    query_vec = np.array(res_query['embedding'], dtype=np.float32).reshape(1, -1)
    
    # 3. Busca FAISS
    D, I = vector_index.search(query_vec, k=8)
    context_confluence = "\n---\n".join([documents[i]['text'] for i in I[0] if i != -1])
    
    # 4. Montagem Multimodal
    contents = []
    if uploaded_file:
        image_bytes = uploaded_file.read()
        contents.append({"inline_data": {"data": base64.b64encode(image_bytes).decode("utf-8"), "mime_type": uploaded_file.type}})
        contents.append("Uma imagem/log foi anexada. Analise tecnicamente.")

    full_prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"PERGUNTA DO USUÁRIO: {user_query}\n\n"
        f"--- CONTEXTO JIRA ---\n{jira_context}\n\n"
        f"--- CONTEXTO CONFLUENCE ---\n{context_confluence}"
    )
    contents.append(full_prompt)
    
    try:
        response = client.models.generate_content(model="gemini-1.5-flash", contents=contents)
        return response.text
    except Exception as e:
        return f"Erro na geração: {e}"

# --- 3. INTERFACE STREAMLIT (SUA INTERFACE ORIGINAL) ---

st.set_page_config(
    page_title="Agente de Suporte para procedimento e correção de problemas relacionados ao sistema",
    layout="wide"
)

st.title("Agente de Suporte (Rodrigo GPT🤓🐋)")
st.markdown("Olá!! Sou Rodrigo GPT, seu assistente de suporte para consulta de dúvidas e procedimento.")
st.markdown("---")

# Expander de Imagem Original
with st.expander("🖼️ Clique aqui para enviar uma Evidência, Captura de Tela ou Log para leitura"):
    uploaded_file = st.file_uploader(
        "Selecione a imagem (PNG, JPG, JPEG) para o agente analisar:",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Imagem Carregada com Sucesso", width=200)

# Lógica de Inicialização da Base (Com o Spinner Original)
if 'vector_index' not in st.session_state:
    with st.spinner("😵 Aguarde um momento, estou acessando minha base de conhecimento para te auxiliar nas suas questões..."):
        knowledge_base = []
        for space in SPACE_KEYS:
            data = busca_conteudo_confluence(space)
            if data:
                knowledge_base.extend(data)
            else:
                st.warning(f"⚠️ Espaço {space} vazio ou sem acesso.")

        if knowledge_base:
            idx, docs = create_vector_store(knowledge_base)
            st.session_state['vector_index'] = idx
            st.session_state['documents'] = docs
            st.success(f"✅ Base de Conhecimento Carregada! Total: {len(knowledge_base)} páginas.")
        else:
            st.error("❌ Falha crítica: Nenhuma base carregada.")
            st.stop()

# Histórico de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de Chat
if user_query := st.chat_input("Qual a sua dúvida ou procedimento?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("🤖 Buscando e analisando procedimentos..."):
        response = gerar_resposta_rag(
            user_query, 
            st.session_state['vector_index'], 
            st.session_state['documents'], 
            uploaded_file
        )
        
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})