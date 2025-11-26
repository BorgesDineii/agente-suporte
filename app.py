import streamlit as st
import requests
import google.genai
import os
from bs4 import BeautifulSoup
import base64
import json

api_key = "xxxx"
ATLASSIAN_USER = "valdinei.borges@e-deploy.com.br"
ATLASSIAN_TOKEN = "xxxx-HeQXotkpCj3tN1LzABhvv0MaI2GkZqDoTII98=FA994E2B"
CONFLUENCE_URL = "https://edeploy.atlassian.net"

CONFLUENCE_URL = "https://edeploy.atlassian.net"
USER_EMAIL = "valdinei.borges@e-deploy.com.br"
API_TOKEN = "xxx-HeQXotkpCj3tN1LzABhvv0MaI2GkZqDoTII98=FA994E2B"
SPACE_KEY = "SPOS2"

if not all([CONFLUENCE_URL, USER_EMAIL, API_TOKEN]):
    print("ERRO: Configure as variaveis de ambiente (ATLASSIAN_USER, ATLASSIAN_TOKEN e CONFLUENCE_URL).")
    exit()

def get_auth_headers():
    """
    Cria um cabeçalho de autenticação necessaria (Basic Auth).
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


try:
    client = google.genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Erro ao iniciar o cliente Google GenAI: {e}")
    st.stop()

st.set_page_config(
    page_title="Agente de Suporte para procedimento e correção de problemas relacionados ao sistema",
    layout="wide"
)


st.title("Agente de Suporte (Rodrigo GPT🤓🐋)")
st.markdown("Olá!! Sou seu Rodrigo GPT, seu assistente de suporte para consulta de dúvidas e procedimento.")
st.markdown("---")


perfil_text = st.text_area(
    "Me informe sua dúvida, problema ou procedimento que eu possa te ajudar!!",
    height=250,
    placehoder="Ex:'Estou com problemas entender o relatorio de 'Dispersão detalhada' pode me explicar como funciona?'"
)

def generate_assistent_prompt(user_text):
    """
    O Agente de Suporte é um chatbot formal, objetivo e preciso, criado para auxiliar na resolução de problemas internos utilizando exclusivamente informações verificadas na Base de Conhecimento (Confluence), chamados, NDP (Novas Deamandas POS), OXAP (Operações x Atendimentos x Produtos) e tickets existentes na plataforma Jira.
    """
    system_prompt = """
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
    """

    user_prompt = f"O texto do questionamento a ser analisado é o seguinte:\n\n---\n{user_text}\n---"

    return system_prompt, user_prompt


def get_respondendo_pergunta(system_prompt, user_prompt):
    """
    Função de chamada à API adapstada para Gemini com o system_prompt embutido
    """

