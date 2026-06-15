# Imersão IA — Alura + Santander 2026
## Códigos das Lives | Semanas 05–07

---

## Aviso sobre os PDFs de NPS

Na live eu mencionei que vocês precisariam **gerar os PDFs de exemplo** ou **adaptar os códigos para ler direto do CSV**. Mas para facilitar, já criei os PDFs fictícios com dados de exemplo e disponibilizei aqui mesmo no repositório, na pasta `NPS/`.

Ou seja: **não é necessário fazer nada extra** — é só configurar o ambiente virtual, instalar as bibliotecas e rodar os scripts normalmente.

---

## O que é um ambiente virtual?

Quando você instala Python no seu computador, ele funciona de forma global — todas as bibliotecas instaladas ficam num só lugar. Isso causa problemas quando projetos diferentes precisam de versões diferentes da mesma biblioteca.

Um **ambiente virtual** (`.venv`) resolve isso criando uma "bolha" isolada para cada projeto: as bibliotecas instaladas dentro dela não interferem no restante do sistema, e vice-versa.

Pense assim: é como ter uma instalação de Python exclusiva para esse projeto, sem bagunçar nada que já está no seu computador.

---

## Pré-requisito: Python 3.11 ou superior

Verifique sua versão com:

```bash
python3 --version
```

Se precisar instalar ou atualizar, acesse [python.org/downloads](https://www.python.org/downloads/).

---

## Passo a passo: configurando o ambiente

### 1. Clone o repositório (se ainda não fez isso)

```bash
git clone https://github.com/elainecro/ImersaoIA_Alura_Santander_Maio2026_Lives.git
cd ImersaoIA_Alura_Santander_Maio2026_Lives
```

### 2. Entre na pasta dos códigos

```bash
cd code
```

### 3. Crie o ambiente virtual

```bash
python3 -m venv .venv
```

Isso cria uma pasta oculta chamada `.venv` dentro de `code/`. Ela contém uma cópia isolada do Python só para esse projeto.

### 4. Ative o ambiente virtual

O comando de ativação é diferente dependendo do seu sistema operacional:

**macOS e Linux:**
```bash
source .venv/bin/activate
```

**Windows (Prompt de Comando):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativo, você verá `(.venv)` no início da linha do terminal — esse é o sinal de que está funcionando:

```
(.venv) $
```

> **Dica:** sempre que abrir um terminal novo para trabalhar nesse projeto, lembre de ativar o ambiente virtual novamente. Ele não fica ativo automaticamente.

### 5. Instale as bibliotecas

Com o ambiente virtual ativo, instale tudo de uma vez:

```bash
pip install -r requirements.txt
```

Esse arquivo lista todas as bibliotecas com as versões exatas usadas nas lives — isso garante que o código vai rodar igual no seu computador.

> O processo pode demorar alguns minutos na primeira vez. É normal.

---

## Configurando a chave de API

Todos os scripts precisam de uma chave da API do Groq. Se você ainda não tem uma, crie gratuitamente em [console.groq.com](https://console.groq.com).

Na pasta `code/`, copie o arquivo de exemplo e preencha com sua chave:

```bash
cp .env.example .env
```

Abra o arquivo `.env` e substitua o valor:

```
GROQ_API_KEY=gsk_sua_chave_aqui
```

---

## Rodando os exemplos

Com o ambiente virtual ativo e o `.env` configurado, rode qualquer script com:

```bash
python nome_do_script.py
```

### Scripts disponíveis

| Script | Descrição | Live |
|---|---|---|
| `agente_reuniao.py` | Resume uma reunião a partir de uma transcrição | Semana 05 |
| `analisador_feedback.py` | Analisa comentários de NPS com classificação por sentimento | Semana 05 |
| `analisador_nps_langchain.py` | Analisa PDFs de NPS usando LangChain e chains encadeadas | Semana 07 |
| `analisador_nps_crew.py` | Analisa PDFs de NPS com uma equipe de agentes CrewAI | Semana 07 |
| `analisador_nps_pydantic.py` | Analisa PDFs de NPS com PydanticAI e output tipado | Semana 07 |

### Exemplos de uso

```bash
# resumo de reunião
python agente_reuniao.py

# análise de feedback/NPS via CSV
python analisador_feedback.py

# análise dos PDFs de NPS (os três fazem a mesma coisa, com abordagens diferentes)
python analisador_nps_langchain.py
python analisador_nps_crew.py
python analisador_nps_pydantic.py
```

---

## Estrutura do projeto

```
github/
├── NPS/                        # PDFs fictícios de pesquisa de satisfação (Aulas 1-6)
├── code/
│   ├── .env.example            # modelo do arquivo de configuração da API
│   ├── requirements.txt        # todas as bibliotecas necessárias
│   ├── reuniao_teste.txt       # transcrição fictícia de reunião (input do agente_reuniao.py)
│   ├── comentarios_nps_completo.csv  # comentários fictícios de NPS (input do analisador_feedback.py)
│   ├── agente_reuniao.py
│   ├── analisador_feedback.py
│   ├── analisador_nps_langchain.py
│   ├── analisador_nps_crew.py
│   └── analisador_nps_pydantic.py
└── README.md
```

---

## Problemas comuns

**"command not found: python3"**
Instale o Python em [python.org/downloads](https://www.python.org/downloads/) e reinicie o terminal.

**"No module named ..."**
Você esqueceu de ativar o ambiente virtual antes de rodar o script. Rode `source .venv/bin/activate` (Mac/Linux) ou o equivalente para Windows.

**"GROQ_API_KEY not found" ou erro de autenticação**
Verifique se o arquivo `.env` existe na pasta `code/` e se a chave está correta (sem espaços, sem aspas extras).

**Erro ao instalar no Windows com PowerShell**
Se aparecer erro de permissão ao ativar o ambiente, rode antes:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
