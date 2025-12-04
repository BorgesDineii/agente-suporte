## 📚 README: 
Agente de Suporte Rodrigo GPT (Confluence RAG & JIRA Integration)

Olá! Bem-vindo ao projeto do Rodrigo GPT, um assistente de suporte inteligente desenvolvido em Python utilizando Streamlit para a interface e a Gemini API (Google GenAI) para processamento de linguagem natural e RAG (Retrieval-Augmented Generation).Este agente é capaz de consultar a Base de Conhecimento do Confluence da e-Deploy, realizar análises multimodais de imagens e logs, e será integrado à API do JIRA para consulta e gestão de tickets.
---
🚀 Funcionalidades PrincipaisRAG com Confluence: Busca procedimentos e informações na Base de Conhecimento (Spaces definidos).
- Análise Multimodal: Capacidade de analisar capturas de tela, logs ou imagens para identificar erros ou procedimentos.
- Histórico de Chat: Mantém o contexto da conversa.
- Integração JIRA (Em Breve): Buscar tickets relacionados à dúvida do usuário para fornecer respostas mais rápidas e vinculadas a soluções existentes.
---
🛠️ Configuração e InstalaçãoSiga os passos abaixo para configurar e rodar o Rodrigo GPT em sua máquina.
1. Pré-requisitos
   Certifique-se de ter o Python (3.9+) instalado em seu ambiente.

3. Clonar o RepositórioAbra seu terminal ou prompt de comando e clone o projeto do GitHub:

```
git clone https://docs.github.com/pt/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
cd [Nome da Pasta do Projeto]
````

3. Criar Ambiente Virtual (Recomendado)Crie e ative um ambiente virtual para isolar as dependências do projeto:
Cria o ambiente virtual
```
python -m venv venv
```
# Ativação (Windows)
```
.\venv\Scripts\activate
```

# Ativação (Linux/macOS)
```
source venv/bin/activate
```

4. Instalar DependênciasInstale todas as bibliotecas necessárias listadas no seu arquivo requirements.txt:
```
pip install -r requirements.txt
```

🔑 Configuração de Credenciais
-

O projeto requer chaves de acesso para o Gemini API e para a API do Atlassian (Confluence/JIRA).Para maior segurança, o ideal é usar Variáveis de Ambiente ou um arquivo .env, mas no seu código, elas estão definidas diretamente. 
-
ATENÇÃO: Mantenha estas chaves em segredo.
-
Credenciais Requeridas: ServiçoVariável no Script (app.py)
-
Onde ObterGemini API
-
api_key Google AI Studio (antigo Google Colab)-
Atlassian UserATLASSIAN_USER / USER_EMAIL - Seu e-mail Atlassian.Atlassian Token ATLASSIAN_TOKEN / API_TOKEN - Token de API criado no Atlassian.Confluence URLCONFLUENCE_URL - URL base da sua instância 
-
(ex: https://nome.atlassian.net).
-
# ▶️ Como Executar o ProjetoCom o ambiente virtual ativado e as credenciais configuradas, execute o aplicativo Streamlit:
```
streamlit run seu_script_principal.py
```
--
(Substitua seu_script_principal.py pelo nome do seu arquivo principal, provavelmente app.py)
-
O Streamlit irá abrir o aplicativo em seu navegador padrão.
-
# 🔄 Comando de Atualização (Pull)
---
Para garantir que você esteja sempre rodando a versão mais recente do projeto diretamente do repositório, utilize o comando 
```
git pull
```
Este comando baixa e mescla as alterações mais recentes do repositório remoto para o seu diretório local.
--
Comando para Atualizar o Projeto:
```
git pull origin main
```
## Explicação:
---
- git pull: Inicia o processo de atualização.
- origin: É o apelido padrão para o seu repositório no GitHub.
- main: É o nome do branch principal do seu projeto. (Se você usa outro nome como master, use-o no lugar de main).
Execute este comando dentro da pasta do projeto (cd [Nome da Pasta do Projeto]) sempre que souber que há novas atualizações no GitHub.
-
# 💡 Próxima Etapa: Implementação JIRA integração com a API do JIRA será implementada em breve. Essa funcionalidade visa enriquecer as respostas do agente, fornecendo contexto em tempo real sobre tickets de suporte relacionados.
