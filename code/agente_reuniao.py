"""
Demo 2 — Agente de Resumo de Reunião
Semana 05 | Imersão Alura + Santander | Especialista em IA

Fluxo:
  reuniao_teste.txt → resumir_reuniao() via Groq → JSON estruturado → reuniao_resumo.md

Requer: pip install groq python-dotenv
Configurar: arquivo .env com GROQ_API_KEY=gsk_...
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Cliente ──────────────────────────────────────────────────────
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ── System Prompt ────────────────────────────────────────────────
SYSTEM_REUNIAO = """
Você é um assistente de produtividade corporativa.
Dada a transcrição de uma reunião, retorne APENAS um JSON com:
  resumo:          parágrafo de 3-4 frases com os pontos principais
  decisoes:        lista de strings com as decisões tomadas
  proximos_passos: lista de objetos {acao, responsavel, prazo}

Regras:
  - Se algum campo não for identificável, retorne lista vazia
  - proximos_passos.prazo: use "não definido" se não mencionado
  - Responda somente com o JSON, sem texto adicional, sem ```json```
""".strip()


# ── Funções ──────────────────────────────────────────────────────
def resumir_reuniao(transcricao: str) -> dict:
    """Recebe a transcrição e retorna um dicionário com resumo, decisões e próximos passos."""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_REUNIAO},
            {"role": "user",   "content": transcricao},
        ],
        temperature=0.1,  # mínimo para máxima consistência
    )
    return json.loads(resp.choices[0].message.content)


def salvar_markdown(resultado: dict, nome_arquivo: str):
    """Formata o resultado como Markdown e salva em arquivo."""
    linhas = [
        "# Resumo da Reunião\n",
        f"## Resumo\n\n{resultado['resumo']}\n",
        "## Decisões\n",
    ]

    for decisao in resultado.get("decisoes", []):
        linhas.append(f"- {decisao}")

    linhas.append("\n## Próximos Passos\n")

    for passo in resultado.get("proximos_passos", []):
        acao       = passo.get("acao", "")
        responsavel = passo.get("responsavel", "?")
        prazo      = passo.get("prazo", "não definido")
        linhas.append(f"- **{acao}** — {responsavel} *(prazo: {prazo})*")

    with open(nome_arquivo, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

    print(f"Resumo salvo em '{nome_arquivo}'")


# ── Execução ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Lê a transcrição
    with open("reuniao_teste.txt", encoding="utf-8") as f:
        transcricao = f.read()

    print("Processando transcrição...\n")

    # Analisa
    resultado = resumir_reuniao(transcricao)

    # Mostra no terminal
    print("── Decisões ─────────────────────────────")
    for d in resultado.get("decisoes", []):
        print(f"  • {d}")

    print("\n── Próximos Passos ──────────────────────")
    for p in resultado.get("proximos_passos", []):
        print(f"  • {p['acao']} — {p['responsavel']} ({p['prazo']})")

    # Salva como .md
    salvar_markdown(resultado, "reuniao_resumo.md")
