import streamlit as st
import requests
from google import genai # Mudança para o import correto
import os
from bs4 import BeautifulSoup
import base64
import json
import numpy as np # Adicionado
import faiss # Adicionado
from langchain_text_splitters import RecursiveCharacterTextSplitter # Adicionado

# --- 1. CONFIGURAÇÕES E VARIÁVEIS ---
# ATENÇÃO: É recomendado usar st.secrets ou variáveis de ambiente para estas chaves.
# Deixei como variáveis diretas para fins de restauração, mas remova antes de fazer commit!
api_key = "xxxx"
ATLASSIAN_USER = "valdinei.borges@e-deploy.com.br"
ATLASSIAN_TOKEN = "xxxx-HeQXotkpCj3tN1LzABhvv0MaI2GkZqDoTII98=FA994E2B"
CONFLUENCE_URL = "https://edeploy.atlassian.net"

CONFLUENCE_URL = "https://edeploy.atlassian.net"
USER_EMAIL = "valdinei.borges@e-deploy.com.br"
API_TOKEN = "xxx-HeQXotkpCj3tN1LzABhvv0MaI2GkZqDoTII98=FA994E2B"
SPACE_KEY = "SPOS2"

if not all([CONFLUENCE_URL, USER_EMAIL, API_TOKEN]):
    st.error("ERRO: Configure as variaveis de ambiente (ATLASSIAN_USER, ATLASSIAN_TOKEN e CONFLUENCE_URL).")
    st.stop()

# Inicialização do cliente Gemini
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Erro ao iniciar o cliente Google GenAI: {e}")
    st.stop()


# --- 2. FUNÇÕES DE INGESTÃO E LIMPEZA DE DADOS ---

def get_auth_headers():
    """
    Cria um cabeçalho de autenticação (Basic Auth).
    """
    auth_string = f"{USER_EMAIL}:{API_TOKEN}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    return{
        "Authorization": f"Basic {encoded_auth}",
        "Accept":"application/json"
    }

def limpando_html_content(html_content):
    """
    Remove tags HTML e limpa o texto do conteudo do Confluence.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    for tag in soup(["nav", "header", "footer", "style", "script"]):
        tag.decompose()

    return soup.get_text(separator=' ', strip=True)

def busca_conteudo_confluence(space_key):
    """
    Busca todas as páginas de um Space Key e extrai o conteúdo limpo.
    """
    # Usando o método mais robusto (auth=(email, token)) para evitar problemas de Base64
    auth_credentials = (USER_EMAIL.strip(), API_TOKEN.strip()) 
    headers = {"Accept":"application/json"}
    
    url = f"{CONFLUENCE_URL}/wiki/rest/api/content?spaceKey={space_key}&expand=body.storage&limit=25"
    clean_knowledge_base = []

    try:
        # CORREÇÃO: Usando 'auth' no requests para autenticação direta
        response = requests.get(url, headers=headers, auth=auth_credentials) 
        response.raise_for_status()

        data = response.json()
        st.success(f"✅ Conectado ao Confluence. Encontradas {len(data.get('results',[]))} páginas.")
        
        for page in data.get('results', []):
            title = page.get('title')
            html_content = page.get('body', {}).get('storage', {}).get('value', '')

            if html_content:
                # CORREÇÃO: Alterando a chamada da função para 'limpando_html_content'
                clean_text = limpando_html_content(html_content) 
                clean_knowledge_base.append({
                    "title":title,
                    "text":clean_text
                })
        return clean_knowledge_base

    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Erro HTTP ao conectar: {e}. Verifique seu Token.")
        return None
    except Exception as e:
        st.error(f"❌ Erro desconhecido: {e}")
        return None
        
def create_vector_store(knowledge_base, client):
    """
    Divide o texto em chunks, gera embeddings e cria o índice vetorial FAISS.
    """
    documents = []
    
    # 1. Chunking (Divisão do texto)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, 
        length_function=len 
    )

    for item in knowledge_base:
        chunks = text_splitter.split_text(item['text'])
        for chunk in chunks:
            documents.append({
                "title": item['title'],
                "chunk": chunk,
                "text": chunk 
            })
    
    # 2. Embedding (Geração de Vetores)
    texts = [doc["text"] for doc in documents]
    
    try:
        response = client.models.embed_content(
            model='text-embedding-004', 
            contents=texts 
        )

        raw_embeddings_list = [item.values for item in response.embeddings]

        # CORREÇÃO APLICADA AQUI: Notação de Ponto
        embeddings = np.array(raw_embeddings_list, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

    except Exception as e:
        st.error(f"❌ Erro no Embedding do Gemini: {e}")
        return None, None

    # 3. FAISS (Criação do Índice Vetorial)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    return index, documents


# --- 3. FUNÇÃO DE RESPOSTA RAG ---

def gerar_resposta_rag(user_query, vector_index, documents, client):
    """
    Busca o contexto relevante no índice FAISS e usa o Gemini para gerar uma resposta.
    """
    # 1. Recuperação (Retrieval)
    
    # Cria o embedding da pergunta do usuário
    query_embedding_response = client.models.embed_content(
        model='text-embedding-004',
        contents=[user_query]
    )

    raw_query_vector = query_embedding_response.embeddings[0].values

    # CORREÇÃO APLICADA AQUI: Notação de Ponto
    query_embedding = np.array(raw_query_vector, dtype=np.float32)

    # Busca os 3 chunks mais relevantes
    D, I = vector_index.search(query_embedding.reshape(1, -1), k=3) 
    
    # Constrói o contexto com o texto dos chunks recuperados
    retrieved_texts = [documents[i]['text'] for i in I[0] if i != -1]
    
    if not retrieved_texts:
        return "Desculpe meu nobre, não encontrei informações relevantes na Base de Conhecimento de Suporte"

    context = "\n---\n".join(retrieved_texts)

    # 2. Prompt Engineering
    system_instruction = (
    """
        O Agente de Suporte é um chatbot formal, objetivo e preciso, criado para auxiliar na resolução de problemas internos utilizando exclusivamente informações verificadas na Base de Conhecimento (Confluence), chamados, NDP (Novas Deamandas POS), OXAP (Operações x Atendimentos x Produtos) e tickets existentes na plataforma Jira.
   
    🔹 Regras Gerais de Atendimento
    1. Pergunta inicial obrigatória
    Antes de qualquer resposta, sempre pergunte ao usuário:
    "Qual sistema você está se referindo? MWPOS ou 3S?"
    
    2. Associação de sistemas
    - MWPOS / MWPOS_KDS → Utilizado apenas em lojas BK e BKF.
    - 3S Checkout → Utilizado por todas as demais lojas.

    3. Fontes de informação
    - Procedimentos → Localizados no Confluence.
    - Tickets de desenvolvimento → OXAP e NDP (não são procedimentos, mas podem conter erros e correções relevantes ao problema informado pelo usuário).
    - Chamados → Consultar no Jira para localizar casos semelhantes.

    4. Memória de conversa
    - Armazenar e manter contexto para que seja possível continuar a conversa de onde parou.
    - Retomar pesquisas ou tickets já consultados durante a interação.

    ---

    🔹 Funções e Responsabilidades

    1. Consulta ao Confluence
    - Pesquisar e apresentar apenas procedimentos oficiais.
    - Fornecer instruções passo a passo com clareza.
    - Sempre que possível, incluir links diretos para documentos, manuais e anexos.

    2. Vinculação de Chamados
    - Procurar chamados anteriores com problemas semelhantes.
    - Apresentar a solução adotada e o número/ticket para referência.

    3. Integração com OXAP e NDP
    - Localizar OXAPs e NDPs relacionados ao problema.
    - Analisar o conteúdo, não apenas o título, para compreender erros e correções.
    - Exibir o resumo ou conteúdo completo, quando necessário.
    - estar sempre atualizado referente a OXAP e NDP do jira.

    4. Análise de Erros e Problemas Recorrentes
    - Identificar erros já registrados em chamados, OXAPs ou NDPs anteriores.
    - Informar a causa provável e o procedimento adotado para correção.
    - Garantir que a solução seja comunicada para manter todos cientes.

    ---

    🔹 Padrão de Resposta
    - Linguagem: Formal, clara e sem gírias.
    - Estrutura:
    1. Descrição do problema
    2. Possíveis causas
    3. Passo a passo da solução
    4. Links/documentos de apoio
    - Quando não encontrar solução:
    Informar que não foi localizado nenhum procedimento e que a questão será encaminhada ao setor responsável.

    ---

    🔹 Restrições Importantes
    - Não inventar procedimentos ou informações.
    - Utilizar apenas conteúdo da base oficial (Confluence, Jira, OXAP, NDP).
    - Sempre tentar localizar chamado ou ticket similar antes de responder que não há solução.

    ---

    🔹 Opção de Melhoria
    Caso o assistente não encontre a resposta correta ou não localize um procedimento aplicável, ele deve informar ao usuário o seguinte:
    "Não encontrei um procedimento ou solução para este caso. Por favor, entre em contato com [Seu Nome] pelo Microsoft Teams para que possamos criar, corrigir ou atualizar um procedimento para consultas futuras.
    """)

    prompt = (
        f"INSTRUÇÃO: {system_instruction}\n\n"
        f"CONTEXTO DE PROCEDIMENTO:\n{context}\n\n"
        f"PERGUNTA DO USUÁRIO: {user_query}"
    )

    # 3. Geração
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text

# --- 4. LÓGICA PRINCIPAL DO STREAMLIT ---

st.set_page_config(
    page_title="Agente de Suporte para procedimento e correção de problemas relacionados ao sistema",
    layout="wide"
)

st.title("Agente de Suporte (Rodrigo GPT🤓🐋)")
st.markdown("Olá!! Sou seu Rodrigo GPT, seu assistente de suporte para consulta de dúvidas e procedimento.")
st.markdown("---")

# Removendo st.text_area isolado, pois a entrada de chat é mais eficiente

# Função get_respondendo_pergunta removida pois não era usada

if 'vector_index' not in st.session_state:
    # 1. INGESTÃO/CARREGAMENTO
    with st.spinner("⏳ Carregando ou Ingerindo Base de Conhecimento do Confluence..."):
        
        knowledge_base = busca_conteudo_confluence(SPACE_KEY)
        
        if knowledge_base:
            try:
                vector_index, documents = create_vector_store(knowledge_base, client)
                
                st.session_state['vector_index'] = vector_index
                st.session_state['documents'] = documents
                st.success("✅ Base de Conhecimento Carregada com Sucesso!")
            
            except Exception as e:
                st.error(f"❌ Erro fatal durante o Embedding ou FAISS: {e}")
                st.stop()
        else:
            st.error("❌ Erro fatal: Não foi possível carregar o conteúdo do Confluence.")
            st.stop()

# Garante que as variáveis estejam disponíveis
vector_index = st.session_state['vector_index']
documents = st.session_state['documents']

# 2. LÓGICA DO CHAT

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a entrada do usuário
if prompt := st.chat_input("Pergunte sobre um procedimento ou erro..."):
    
    # CORREÇÃO: st.session_state.messages.append
    st.session_state.messages.append({"role": "user", "content": prompt}) 
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🤖 Buscando e analisando procedimentos..."):
        try:
            full_response = gerar_resposta_rag(prompt, vector_index, documents, client)
        except Exception as e:
            full_response = f"Ocorreu um erro ao gerar a resposta: {e}"

    with st.chat_message("assistant"):
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# O bloco 'if __name__ == "__main__":' foi removido e sua lógica integrada ao Streamlit.
# A função de chat do Streamlit já é o bloco principal de execução.