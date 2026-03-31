import streamlit as st
import requests
from google import genai 
import os
from bs4 import BeautifulSoup
import base64
import json
import numpy as np 
import faiss 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import re # Essencial para buscar chaves JIRA como OXAP-5208

# --- 1. CONFIGURAÇÕES E VARIÁVEIS ---
# ATENÇÃO: É recomendado usar st.secrets ou variáveis de ambiente para estas chaves.
# Deixei como variáveis diretas para fins de restauração, mas remova antes de fazer commit!
<<<<<<< HEAD
API_JIRA = "Aq_zsKNq0N-pC9W6pEAzsUJvAff5n6-Tkxx4Zqm9VLV9PaSWEW1k38hqTaXCLtm5vrUNyoBbIAm7ILksbcCuYDpPVplDRzMrZYNM0vk=BD8EB151"
api_key = "OAhKS_GY"
ATLASSIAN_USER = "valdinei.borges@e-deploy.com.br"
ATLASSIAN_TOKEN = "q0N-pC9W6pEAzsUJvAff5n6-Tkxx4Zqm9VLV9PaSWEW1k38hqTaXCLtm5vrUNyoBbIAm7ILksbcCuYDpPVplDRzMrZYNM0vk=BD8EB151"
CONFLUENCE_URL = "https://edeploy.atlassian.net"
=======
API_JIRA = "CCCD-CDCDCD=BD8EB151"
api_key = "AIzaSyAuqvAA-m7BfEekEjf8NDyo9q8OAhKS_GY"
ATLASSIAN_USER = "valdinei.borges@e-deploy.com.br"
ATLASSIAN_TOKEN = "DCDCDCD-=BD8EB151"
CONFLUENCE_URL = "https://edeploy.atlassian.net"
>>>>>>> 3db5ddc1f034589ca7028a96ef568b745c20bdb4

<<<<<<< HEAD
CONFLUENCE_URL = "https://edeploy.atlassian.net"
USER_EMAIL = "valdinei.borges@e-deploy.com.br"
API_TOKEN = "vvk=BD8EB151"
SPACE_KEY = "SPOS2"

=======
CONFLUENCE_URL = "https://edeploy.atlassian.net"
USER_EMAIL = "valdinei.borges@e-deploy.com.br"
API_TOKEN = "=BD8EB151"
# SPACE_KEY = "SPOS2"

>>>>>>> 3db5ddc1f034589ca7028a96ef568b745c20bdb4
# --- 1. CONFIGURAÇÕES E VARIÁVEIS (Atualizado) ---
# ... (seus imports e outras variáveis)

# Remova o 'SPACE_KEY' singular e use o plural 'SPACE_KEYS'
SPACE_KEYS = ["SPOS2", "SPOS1"] # CORRIGIDO: Nome da variável de iteração

# Sua instrução de sistema DEVE ser global (remova a definição da função gerar_resposta_rag)
SYSTEM_INSTRUCTION = """
Você é o **Rodrigo GPT**, um Agente de Suporte Técnico da E-DEPLOY. Sua função é ser proativo, respeitoso e fornecer soluções e procedimentos claros, baseados estritamente nos contextos fornecidos (JIRA e Confluence).

**REGRAS DE CONDUTA:**
1. **Persona:** Se a pergunta for "quem é voce?", responda: "Olá, sou Rodrigo GPT, um grande fã de churros e comida."
2. **Prioridade na Solução:** A resposta deve ser extraída dos contextos. Não use links externos, EXCETO o link do ticket JIRA específico encontrado (ex: 'Link: https://.../browse/TICKET-123').

**PRIORIDADE DE CONTEXTO E GERAÇÃO:**
1. **Confluence RAG (Primeira Prioridade):** Se o JIRA não fornecer uma solução, utilize **APENAS** o 'CONTEXTO DE PROCEDIMENTO' (Confluence) para gerar um passo a passo.


2. **JIRA (Segunda Prioridade e Foco):**O bloco 'CONTEXTO JIRA PRIORITÁRIO' é sua fonte principal. Seu objetivo é **identificar a similaridade** entre a PERGUNTA do usuário e o **Resumo/Descrição/Logs** dos tickets encontrados.
    * **Ação Obrigatória:** Se encontrar chamados similares, você **DEVE** informar ao usuário que encontrou um chamado anterior com problema semelhante.
    * **Conteúdo e Link:** Apresente um breve resumo da ocorrência similar e forneça **APENAS o Link de Referência** desse chamado para que o usuário faça a análise final. (Não use links de dashboard).
    
3. **Análise Multimodal:** Se uma imagem ou log for anexado, use a análise dedutiva antes de aplicar o RAG.

**LIMITE DE CONHECIMENTO E FALLBACK (Último Recurso):**
1. **Utilize APENAS** as informações contidas nos blocos de contexto.
2. **Se NENHUM contexto (JIRA, Confluence, Imagem) fornecer uma solução clara**, utilize o seguinte fallback **EXCLUSIVO**: "Não encontrei o procedimento solicitado na Base de Conhecimento. Peço que solicite ajuda interna para lhe ajudar com isso. Vou voltar a comer meu Churros."
"""

if not all([CONFLUENCE_URL, USER_EMAIL, API_TOKEN
]):
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
    Busca todas as páginas de um Space Key, extrai o conteúdo limpo e implementa a paginação.
    """
    auth_credentials = (USER_EMAIL.strip(), API_TOKEN.strip()) 
    headers = {"Accept": "application/json"}
    
    # Parâmetros de busca
    MAX_LIMIT = 500 # Limite máximo que queremos baixar (o servidor pode impor um limite menor por requisição, como 25)
    START_PAGINATION = 0
    clean_knowledge_base = []
    
    # st.info(f"Iniciando busca no espaço {space_key}...")

    while True:
        # A API v1 geralmente usa 'start' e 'limit'
        url = (
            f"{CONFLUENCE_URL}/wiki/rest/api/content?spaceKey={space_key}"
            f"&expand=body.storage"
            f"&limit=100" # Usamos um limite seguro por requisição (100)
            f"&start={START_PAGINATION}"
        )

        try:
            response = requests.get(url, headers=headers, auth=auth_credentials) 
            response.raise_for_status()

            data = response.json()
            
            # 1. Processa os resultados desta página/lote
            results = data.get('results', [])
            
            for page in results:
                title = page.get('title')
                html_content = page.get('body', {}).get('storage', {}).get('value', '')

                if html_content:
                    # Chamar limpando_html_content
                    clean_text = limpando_html_content(html_content) 
                    clean_knowledge_base.append({
                        "title": title,
                        "text": clean_text
                    })

            # 2. Verifica a Paginação
            size_of_results = len(results)
            total_size = len(clean_knowledge_base)
            
            # st.info(f"Espaço {space_key}: Páginas encontradas neste lote: {size_of_results}. Total: {total_size}")

            # Se a quantidade de resultados for menor que o limite, ou se já atingimos o total desejado, paramos.
            # Também usamos 'start' + 'size' == 'total' (propriedade que a API geralmente retorna)
            if size_of_results < 100 or total_size >= MAX_LIMIT:
                break

            # Prepara para o próximo lote
            START_PAGINATION += size_of_results
            
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Erro HTTP ao conectar ao Confluence no espaço {space_key}: {e}")
            return None
        except Exception as e:
            st.error(f"❌ Erro desconhecido durante a busca no espaço {space_key}: {e}")
            return None
            
    # st.success(f"✅ Conexão bem-sucedida ao Confluence. Espaço {space_key} carregado.")
    return clean_knowledge_base
        
def create_vector_store(knowledge_base, client):
    """
    Divide o texto em chunks, gera embeddings e cria o índice vetorial FAISS.
    """
    documents = []
    
    # 1. Chunking (Divisão do texto)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400, 
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

    BATCH_SIZE = 100
    all_raw_embeddings = []

    st.info(f"Gerando embeeedings para {len(texts)} chunks em lotes de {BATCH_SIZE}...")
    st.sidebar.progress(0.0)

    try:
        total_chunks = len(texts)

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            if not batch_texts:
                continue


            response = client.models.embed_content(
                model='text-embedding-004', 
                contents=batch_texts 
            )

            batch_raw_embeddings = [item.values for item in response.embeddings]
            all_raw_embeddings.extend(batch_raw_embeddings)
# Bloco temporário para diagnóstico:
            print("--- MODELOS DISPONÍVEIS NA SUA CHAVE ---")
            for m in client.models.list():
                print(f"Nome: {m.name} - Suporta Embedding: {'embedContent' in m.supported_generation_methods}")
            print("---------------------------------------")
            current_progress = (i + len(batch_texts)) / total_chunks
            st.sidebar.progress(current_progress)

        embeddings = np.array(all_raw_embeddings, dtype=np.float32)

        # raw_embeddings_list = [item.values for item in response.embeddings]

        # CORREÇÃO APLICADA AQUI: Notação de Ponto
        # embeddings = np.array(raw_embeddings_list, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        st.sidebar.progress(1.0)

    except Exception as e:
        # Erro de debugging
        error_message = f"400 INVALID_ARGUMENT. {e}" if "400" in str(e) else str(e)
        print(f"ERRO FATAL NO EMBEDDING: {error_message}")
        st.error(f"❌ Erro no Embedding do Gemini: Falha ao criar vetores. Detalhe: {error_message}")
        return None, None

    # 3. FAISS (Criação do Índice Vetorial)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    return index, documents


# --- 3. FUNÇÃO DE RESPOSTA RAG ---

# --- 3. FUNÇÃO DE RESPOSTA RAG (CORRIGIDA) ---
def busca_chamados_jira(user_query, max_results=3):
    jira_url = f"{CONFLUENCE_URL}/rest/api/3/search/jql" # ENDPOINT CORRIGIDO
    
    # 1. Autenticação Basic Auth em Base64
    auth_string = f"{USER_EMAIL}:{API_JIRA}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # 2. Lógica JQL (Híbrida)
    jira_key_pattern = r'([A-Z]{2,}-\d+)'
    explicit_keys = re.findall(jira_key_pattern, user_query.upper())
    
    jql_parts = []
    jql_parts.append(f'text ~ "{user_query}"')
    
    if explicit_keys:
        key_clauses = [f'key = "{key}"' for key in set(explicit_keys)]
        jql_parts.append(" OR ".join(key_clauses))

    jql_query = f'{jql_parts[0]}'
    if len(jql_parts) > 1:
        jql_query = f'({jql_query}) OR ({jql_parts[1]})'
        
    jql_query = f'{jql_query} ORDER BY updated DESC'
    
    payload = {
        "jql": jql_query,
        "fields": ["key", "summary", "status", "resolution", "issuetype", "description", "comment"],
        "maxResults": max_results
    }
    
    try:
        response = requests.post(jira_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        data = response.json()
        tickets = []
        
        for issue in data.get('issues', []):
            ticket_key = issue['key']
            summary = issue['fields']['summary']
            status = issue['fields']['status']['name']
            resolution = issue['fields'].get('resolution', {}).get('name', 'N/A')
            issue_type = issue['fields']['issuetype']['name']
            description = issue['fields'].get('description', 'N/A')
            comments = issue['fields'].get('comment', {}).get('comments', [])
            last_comment_text = comments[-1].get('body', 'N/A') if comments else 'N/A'
            
            # Monta o contexto que o Gemini vai ler
            ticket_context = (
                f"--- CHAMADO JIRA SIMILAR: {ticket_key} ({issue_type}) ---\n"
                f"Status Atual: {status} (Resolução: {resolution})\n"
                f"Resumo da Ocorrência: {summary}\n"
                f"Descrição/Logs (Amplo Contexto): {description[:500]}...\n" # Limitar para não encher o prompt
                f"Último Comentário Relevante: {last_comment_text[:500]}...\n"
                f"Link de Referência: {CONFLUENCE_URL}/browse/{ticket_key}"
            )
            tickets.append(ticket_context)
            
        return tickets

    except requests.exceptions.HTTPError as e:
        print(f"ERRO JIRA HTTP: Falha na busca ou permissões. {e}")
        return None
    except Exception as e:
        print(f"ERRO JIRA DESCONHECIDO: {e}")
        return None

def gerar_resposta_rag(user_query, vector_index, documents, client, uploaded_file):
    """
    Busca o contexto relevante no índice FAISS e usa o Gemini para gerar uma resposta.
    """
    
    # 1. Recuperação (Retrieval)
    jira_context = "" 
    
    contents = []
    
    # === BUSCA JIRA ===
    # A função busca_chamados_jira retorna uma lista de strings ou None
    jira_tickets = busca_chamados_jira(user_query, max_results=3)
    
    # O bloco de if verifica se a lista não está vazia ou é None
    if jira_tickets and len(jira_tickets) > 0:
        jira_context = "\n\n--- TICKETS JIRA RELACIONADOS ---\n" + "\n---\n".join(jira_tickets)
        print(f"DEBUG: {len(jira_tickets)} tickets JIRA encontrados.")
    else:
        print("DEBUG: Nenhum ticket JIRA relevante encontrado.")
    # ========================
    
    # ... (código de embedding da query, FAISS search e contexto)

    # Cria o embedding da pergunta do usuário
    query_embedding_response = client.models.embed_content(
        model='text-embedding-004',
        contents=[user_query]
    )

    raw_query_vector = query_embedding_response.embeddings[0].values
    query_embedding = np.array(raw_query_vector, dtype=np.float32)

    # Busca os 10 chunks mais relevantes
    D, I = vector_index.search(query_embedding.reshape(1, -1), k=10) 

    print("\n--- FAISS DEBUG ---")
    print(f"Query: {user_query}")
    print(f"Distâncias (D): {D}")
    print(f"Índices (I): {I}")
    print("-------------------\n")

    valid_indices = [i for i in I[0] if i != -1]

    if not valid_indices:
        return "Desculpe, nao encontrei informações relevantes na Base de conhecimento para esta busca."
    
    retrieved_texts = [documents[i]['text'] for i in valid_indices]
    context = "\n---\n".join(retrieved_texts)


    # 2. Montagem Multimodal (CORRIGIDO: Escopo de contents.append())
    
    # 2.1 Adiciona a Imagem (Condicional)
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            
            # Adiciona o objeto da imagem
            contents.append({
                "inline_data": {
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                    "mime_type": uploaded_file.type 
                }
            })
            
            # Adiciona uma instrução textual específica para a imagem
            contents.append(
    "ATENÇÃO: Uma imagem/log foi anexada. Sua tarefa primária é realizar uma **Análise Técnica DEDUTIVA** para identificar o erro ou o procedimento. Utilize o 'CONTEXTO DE PROCEDIMENTO' (RAG) apenas como auxílio secundário. Se você conseguir identificar a causa ou solução pela imagem, ignore o RAG fallback."
)
            
        except Exception as e:
            st.warning(f"Não foi possível processar a imagem: {e}")

    # 2.2 Constrói e Anexa o Prompt RAG Principal (EXECUTADO SEMPRE)
    if jira_context:
        jira_block = f"--- CONTEXTO JIRA PRIORITÁRIO (TICKETS ENCONTRADOS) ---\n{jira_context}\n"
    else:
        jira_block = "--- CONTEXTO JIRA PRIORITÁRIO ---\nNenhum ticket JIRA relevante foi encontrado no momento.\n"
        
    # 2.2.2 Monta o Bloco de Contexto Confluence
    confluence_block = f"--- CONTEXTO DE PROCEDIMENTO ---\n{context}\n"
        
    full_prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"PERGUNTA DO USUÁRIO: {user_query}\n\n"
        # O JIRA VEM ANTES DO CONFLUENCE
        f"{jira_block}\n" 
        f"{confluence_block}"
    )

    contents.append(full_prompt) # <--- ESSA LINHA AGORA ESTÁ FORA DO IF DA IMAGEM
    
    # DEBUG: Para confirmar que a lista não está vazia.
    print(f"DEBUG FINAL: Contents len: {len(contents)}")


    # 3. Geração
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=contents 
        )
        return response.text
        
    except Exception as e:
        print(f"Erro na chamada do Gemini API: {e}")
        return "❌ Desculpe, houve uma falha de comunicação com o serviço de IA. Tente novamente, ou verifique sua chave de API."
# --- 4. LÓGICA PRINCIPAL DO STREAMLIT ---

st.set_page_config(
    page_title="Agente de Suporte para procedimento e correção de problemas relacionados ao sistema",
    layout="wide"
)

st.title("Agente de Suporte (Rodrigo GPT🤓🐋)")
st.markdown("Olá!! Sou Rodrigo GPT, seu assistente de suporte para consulta de dúvidas e procedimento.")
st.markdown("---")
with st.expander("🖼️ Clique aqui para enviar uma Evidência, Captura de Tela ou Log para leitura"):
    uploaded_file = st.file_uploader(
        "Selecione a imagem (PNG, JPG, JPEG) para o agente analisar:",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Imagem Carregada com Sucesso", width=200)
#uploaded_file = st.file_uploader(
#    "Ou envie uma captura de tela para análise do erro:",
#    type=["png", "jpg", "jpeg"]
#)
#with st.sidebar.expander("📎 Anexar Imagem para Análise"):
#    uploaded_file = st.file_uploader(
#        "Selecione a captura de tela (PNG, JPG):",
#        type=["png", "jpg", "jpeg"]
#    )

#user_query = st.chat_input("Qual a sua dúvida ou procedimento?")



# Função get_respondendo_pergunta removida pois não era usada

if 'vector_index' not in st.session_state:
    knowledge_base = []
    log_messages = []
    
    # 1. Mensagem amigável para o usuário
    with st.spinner("😵 Aguarde um momento, estou acessando minha base de conhecimento para te auxiliar nas suas questões..."):
        
        # Itera sobre CADA espaço
        for space in SPACE_KEYS:
            
            # Não exibe o st.info para o usuário final, apenas registra para debug
            log_messages.append(f"Iniciando busca no espaço: {space}") 
            
            space_data = busca_conteudo_confluence(space)
            
            if space_data:
                knowledge_base.extend(space_data)
            else:
                # Mantém o warning visível para o desenvolvedor em caso de falha
                st.warning(f"⚠️ Não foi possível carregar o conteúdo do espaço {space}. Verifique permissões.") 

        if knowledge_base:
            try:
                # 2. Criação do Vector Store
                vector_index, documents = create_vector_store(knowledge_base, client)
                
                st.session_state['vector_index'] = vector_index
                st.session_state['documents'] = documents
                
                # Mensagem final de sucesso (aparece após o spinner desaparecer)
                st.success(f"✅ Base de Conhecimento Carregada com Sucesso! Total de páginas: {len(knowledge_base)}.")
            
            except Exception as e:
                st.error(f"❌ Erro fatal durante o Embedding ou FAISS: {e}")
                st.stop()
        else:
            st.error("❌ Erro fatal: Nenhuma base de conhecimento pôde ser carregada.")
            st.stop()

# Garante que as variáveis estejam disponíveis
vector_index = st.session_state['vector_index']
documents = st.session_state['documents']

# 2. LÓGICA DO CHAT

# --- 2. LÓGICA DO CHAT (CORRIGIDA) ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico de mensagens PRIMEIRO
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a entrada do usuário e inicia a lógica POR SEGUNDO
if user_query := st.chat_input("Qual a sua dúvida ou procedimento?"):
    
    # 1. Adicionar a pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Exibir a pergunta do usuário no chat (sem duplicidade, pois o histórico já foi exibido)
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("🤖 Buscando e analisando procedimentos..."):
        try:
            # 3. Gerar a resposta RAG
            full_response = gerar_resposta_rag(
                user_query, 
                vector_index, 
                documents, 
                client,
                uploaded_file
            )
        except Exception as e:
            full_response = f"❌ Ocorreu um erro ao gerar a resposta: {e}"

    # 4. Exibir a resposta do assistente e salvar no histórico
    with st.chat_message("assistant"):
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# Captura a entrada do usuário
#if prompt := st.chat_input("Pergunte sobre um procedimento ou erro..."):
    
    # CORREÇÃO: st.session_state.messages.append
#    st.session_state.messages.append({"role": "user", "content": prompt}) 
#    
#    with st.chat_message("user"):
#        st.markdown(prompt)

#    with st.spinner("🤖 Buscando e analisando procedimentos..."):
#        try:
#            full_response = gerar_resposta_rag(prompt, vector_index, documents, client)
#        except Exception as e:
#            full_response = f"Ocorreu um erro ao gerar a resposta: {e}"

#    with st.chat_message("assistant"):
#        st.markdown(full_response)
#        st.session_state.messages.append({"role": "assistant", "content": full_response})


# O bloco 'if __name__ == "__main__":' foi removido e sua lógica integrada ao Streamlit.
# A função de chat do Streamlit já é o bloco principal de execução.