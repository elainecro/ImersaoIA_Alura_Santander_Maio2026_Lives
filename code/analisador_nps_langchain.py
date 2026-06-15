# =============================================================================
# ANALISADOR DE NPS COM LANGCHAIN
# Imersão Alura + Santander 2026 — Semana 07
#
# pip install langchain-core langchain-groq pdfplumber
# export GROQ_API_KEY="sua_chave"
# =============================================================================

import os
import glob
import time
import pdfplumber
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# ── configuração ──────────────────────────────────────────────────────────────

PASTA_NPS = os.path.join(
    os.path.dirname(__file__),   # pasta onde este script está
    "..", "..", "NPS"            # sobe dois níveis e entra em NPS/
)
MODELO = "llama-3.1-8b-instant"


# ── leitura dos PDFs ──────────────────────────────────────────────────────────

def ler_pdf(caminho: str) -> str:
    """Extrai o texto completo de um arquivo PDF."""
    with pdfplumber.open(caminho) as pdf:
        paginas = [pagina.extract_text() or "" for pagina in pdf.pages]
    return "\n".join(paginas)


def carregar_todos_os_nps() -> dict[str, str]:
    """
    Lê todos os PDFs da pasta NPS.
    Retorna um dicionário {nome_do_arquivo: texto_extraído}.
    """
    padrao = os.path.join(PASTA_NPS, "*.pdf")
    arquivos = sorted(glob.glob(padrao))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum PDF encontrado em: {PASTA_NPS}")

    print(f"\n📂 {len(arquivos)} arquivo(s) encontrado(s):\n")
    textos = {}
    for caminho in arquivos:
        nome = os.path.basename(caminho)
        texto = ler_pdf(caminho)
        textos[nome] = texto
        print(f"   ✓ {nome} ({len(texto)} caracteres)")

    return textos


# ── chains LangChain ──────────────────────────────────────────────────────────

def criar_chains(llm: ChatGroq):
    """
    Cria e retorna as duas chains.

    CHAIN 1 — extração
    Recebe o texto bruto de um PDF de NPS e extrai:
    - nota NPS (número de 0–100)
    - comentários dos alunos

    CHAIN 2 — análise
    Recebe os comentários extraídos e gera um relatório
    com pontos positivos, críticas e recomendação.
    """

    # ── chain 1: extrair dados estruturados do texto bruto ────────────────────
    prompt_extracao = PromptTemplate.from_template("""
Você receberá o texto extraído de um PDF de NPS de uma aula de IA.

Seu trabalho é identificar e retornar:
1. O score NPS (número entre 0 e 100, geralmente aparece como "NPS: XX" ou isolado)
2. Todos os comentários textuais dos alunos (as frases e parágrafos de feedback)

Ignore números de notas de estrelas, datas, nomes de turma e cabeçalhos.
Foque apenas nos comentários escritos pelos alunos.

Formato de saída:
NPS: [número]
COMENTÁRIOS:
- [comentário 1]
- [comentário 2]
...

TEXTO DO PDF:
{texto}
""")

    # ── chain 2: gerar análise a partir dos comentários ───────────────────────
    prompt_analise = PromptTemplate.from_template("""
Você é especialista em análise de satisfação de cursos de tecnologia.

Com base nos comentários abaixo extraídos de pesquisas NPS, gere uma análise clara e objetiva.

COMENTÁRIOS:
{comentarios}

Sua análise deve conter:

📊 NPS DA AULA: {nps}

✅ PONTOS POSITIVOS (top 3 mais mencionados):
1.
2.
3.

⚠️ CRÍTICAS E SUGESTÕES (top 3 mais relevantes):
1.
2.
3.

💡 INSIGHT INESPERADO:
(algo que aparece nos feedbacks mas que pode ser facilmente ignorado)

🎯 RECOMENDAÇÃO PRIORITÁRIA:
(uma ação concreta para a próxima aula)
""")

    # ── montando as chains com o operador pipe ( | ) ──────────────────────────
    # PromptTemplate → LLM → StrOutputParser
    # O StrOutputParser converte o objeto AIMessage em string pura
    chain_extracao = prompt_extracao | llm | StrOutputParser()
    chain_analise  = prompt_analise  | llm | StrOutputParser()

    return chain_extracao, chain_analise


# ── execução principal ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ANALISADOR DE NPS — LANGCHAIN")
    print("=" * 60)

    # inicializa o LLM (temperatura baixa = análise mais consistente)
    llm = ChatGroq(model=MODELO, temperature=0.2)

    # cria as chains
    chain_extracao, chain_analise = criar_chains(llm)

    # carrega os PDFs
    textos_nps = carregar_todos_os_nps()

    resultados = []

    for nome_arquivo, texto in textos_nps.items():
        print(f"\n{'─' * 60}")
        print(f"📄 Processando: {nome_arquivo}")
        print("─" * 60)

        inicio = time.time()

        # CHAIN 1: extrai NPS e comentários do texto bruto
        print("⏳ Chain 1: extraindo dados do PDF...")
        dados_extraidos = chain_extracao.invoke({"texto": texto})

        # separa o NPS dos comentários para passar para a chain 2
        linhas = dados_extraidos.split("\n")
        nps_linha = next((l for l in linhas if l.startswith("NPS:")), "NPS: não identificado")
        nps_valor = nps_linha.replace("NPS:", "").strip()

        # CHAIN 2: gera análise a partir dos dados extraídos
        print("⏳ Chain 2: gerando análise e recomendações...")
        analise = chain_analise.invoke({
            "comentarios": dados_extraidos,
            "nps": nps_valor
        })

        duracao = time.time() - inicio
        print(f"\n✅ Concluído em {duracao:.1f}s\n")
        print(analise)

        resultados.append({
            "arquivo": nome_arquivo,
            "nps": nps_valor,
            "analise": analise
        })

    # resumo final com todas as aulas
    print(f"\n{'=' * 60}")
    print("  RESUMO CONSOLIDADO")
    print("=" * 60)

    prompt_consolidado = PromptTemplate.from_template("""
Você tem as análises de NPS de {n_aulas} aulas diferentes de um curso de IA.

{analises}

Faça um resumo executivo consolidado com:
- Tendência geral do NPS ao longo das aulas
- Padrão de elogios que se repete
- Padrão de críticas que se repete
- 3 ações prioritárias para melhorar o curso
""")

    chain_consolidada = prompt_consolidado | llm | StrOutputParser()

    analises_texto = "\n\n---\n\n".join(
        f"AULA: {r['arquivo']}\n{r['analise']}" for r in resultados
    )

    print("\n⏳ Gerando consolidado de todas as aulas...")
    resumo = chain_consolidada.invoke({
        "n_aulas": len(resultados),
        "analises": analises_texto
    })
    print(resumo)


if __name__ == "__main__":
    load_dotenv()
    main()
