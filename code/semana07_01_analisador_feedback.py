"""
Demo — Analisador de Feedback / NPS
mersão Alura + Santander | Especialista em IA

Fluxo:
  comentarios_teste.txt → analisar() via Groq → JSON por comentário → analise_feedback.csv

Requer: pip install groq python-dotenv pandas
Configurar: arquivo .env com GROQ_API_KEY=gsk_...
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Cliente ──────────────────────────────────────────────────────
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ── System Prompt ────────────────────────────────────────────────
# Projetado com Claude.ai → testado no Google AI Studio → código aqui
SYSTEM = """
Você é um analista de experiência do cliente.
Dado um comentário de aluno, retorne APENAS um JSON com:
  sentimento:  "positivo" | "negativo" | "neutro"
  categoria:   "conteudo" | "tempo" | "professor" | "tecnico" | "outro"
  resumo:      uma frase com o ponto principal do comentário
  prioridade:  "alta" | "media" | "baixa"

Regra de prioridade:
  alta  → problema impede o aluno de continuar o curso
  media → insatisfação significativa mas não bloqueante
  baixa → elogio, sugestão ou comentário neutro

Responda somente com o JSON, sem texto adicional, sem ```json```.
""".strip()


# ── Função principal ─────────────────────────────────────────────
def analisar(comentario: str) -> dict:
    """Analisa um comentário e retorna um dicionário estruturado."""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": comentario},
        ],
        temperature=0.2,  # baixa para consistência no JSON
    )
    return json.loads(resp.choices[0].message.content)


# ── Pipeline ─────────────────────────────────────────────────────
def processar_arquivo(caminho_entrada: str, caminho_saida: str):
    """Lê comentários de um .txt (um por linha) e salva análise em .csv."""
    with open(caminho_entrada, encoding="utf-8") as f:
        comentarios = [l.strip() for l in f.readlines() if l.strip()]

    print(f"\nProcessando {len(comentarios)} comentários...\n")

    resultados = []
    for i, c in enumerate(comentarios, 1):
        resultado = analisar(c)
        resultado["original"] = c
        resultados.append(resultado)
        print(f"[{resultado['prioridade'].upper():5}]  {resultado['resumo']}")

    df = pd.DataFrame(resultados)
    df.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    print(f"\nSalvo: {len(df)} registros em '{caminho_saida}'")
    return df


# ── Execução ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df = processar_arquivo(
        caminho_entrada="comentarios_nps_completo.csv",
        caminho_saida="analise_nps.csv",
    )

    # Resumo por prioridade
    print("\n── Resumo por prioridade ─────────────────")
    print(df["prioridade"].value_counts().to_string())

    print("\n── Resumo por categoria ──────────────────")
    print(df["categoria"].value_counts().to_string())
