# 🤖 Rodrigo GPT: Agente RAG de Suporte Técnico (Confluence)

Este projeto implementa um Chatbot de Geração Aumentada por Recuperação (RAG) utilizando o modelo **Gemini 2.5 Flash** para consultar procedimentos e informações diretamente de uma Base de Conhecimento do Confluence (Atlassian).

O objetivo do agente é fornecer respostas claras e passo a passo, extraindo o conteúdo relevante do contexto e evitando alucinações.

---

## ⚙️ Configuração do Ambiente

### Pré-requisitos

* Python 3.8+
* Acesso à API do Google Gemini.
* Credenciais de acesso à API do Confluence (usuário e token de API).

### 1. Instalação de Dependências

Instale todas as bibliotecas necessárias usando o `requirements.txt`:

```bash
pip install -r requirements.txt
```

2. Configuração de Credenciais
Você deve configurar as seguintes credenciais e chaves de API diretamente no topo do arquivo app.py (ou em variáveis de ambiente, que é a prática recomendada):
