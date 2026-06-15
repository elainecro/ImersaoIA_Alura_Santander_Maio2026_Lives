# =============================================================================
# ANALISADOR DE NPS COM PYDANTICAI — OUTPUT TIPADO E VALIDADO
# Imersão Alura + Santander 2026 — Semana 07
#
# pip install pydantic-ai pdfplumber
# export GROQ_API_KEY="sua_chave"
# =============================================================================

import os
import glob
import asyncio
import pdfplumber
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal

# ── configuração ──────────────────────────────────────────────────────────────

PASTA_NPS = os.path.join(
    os.path.dirname(__file__),
    "..", "NPS"
)

# PydanticAI usa a string 'provedor:modelo' diretamente
# Precisa da variável GROQ_API_KEY no ambiente
MODELO = "groq:llama-3.3-70b-versatile"


# ── schemas de output — o coração do PydanticAI ───────────────────────────────
#
# Aqui está a diferença principal em relação às outras abordagens:
# você define EXATAMENTE o que quer de volta, com tipos Python reais.
# O PydanticAI força o LLM a retornar nesse formato e valida automaticamente.
# Se o LLM errar o formato, ele tenta novamente antes de lançar exceção.

class PontoFeedback(BaseModel):
    """Um ponto de feedback com citação real do aluno."""
    descricao: str = Field(description="O que o aluno disse, em termos gerais")
    citacao: str = Field(description="Trecho real copiado do comentário do aluno")
    frequencia: Literal["único", "alguns", "frequente"] = Field(
        description="Com que frequência esse ponto aparece nos feedbacks"
    )


class AnaliseNPS(BaseModel):
    """
    Resultado completo da análise de NPS de uma aula.
    Todos os campos são validados automaticamente pelo Pydantic.
    """
    nome_aula: str = Field(description="Nome do arquivo de NPS analisado")

    nps_score: float = Field(
        ge=0, le=100,
        description="Score NPS da aula (0 a 100)"
    )

    sentimento_geral: Literal["muito positivo", "positivo", "neutro", "negativo", "crítico"] = Field(
        description="Sentimento predominante nos comentários"
    )

    elogios: list[PontoFeedback] = Field(
        min_length=1, max_length=3,
        description="Top 3 pontos positivos com citações reais"
    )

    criticas: list[PontoFeedback] = Field(
        min_length=1, max_length=3,
        description="Top 3 críticas ou sugestões com citações reais"
    )

    insight_inesperado: str = Field(
        description=(
            "Um padrão ou observação que aparece nos feedbacks mas que "
            "seria fácil de ignorar numa leitura rápida"
        )
    )

    recomendacao_urgente: str = Field(
        description="Uma ação concreta para implementar na próxima aula"
    )

    nivel_engajamento: Literal["baixo", "médio", "alto", "muito alto"] = Field(
        description="Nível de engajamento da turma com base nos comentários"
    )


# ── leitura dos PDFs ──────────────────────────────────────────────────────────

def ler_pdf(caminho: str) -> str:
    with pdfplumber.open(caminho) as pdf:
        paginas = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(paginas)


def carregar_arquivos_nps() -> list[tuple[str, str]]:
    """Retorna lista de (nome_do_arquivo, texto_extraído)."""
    padrao = os.path.join(PASTA_NPS, "*.pdf")
    arquivos = sorted(glob.glob(padrao))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum PDF encontrado em: {PASTA_NPS}")

    print(f"\n📂 {len(arquivos)} arquivo(s) encontrado(s):\n")
    resultado = []
    for caminho in arquivos:
        nome = os.path.basename(caminho)
        texto = ler_pdf(caminho)
        resultado.append((nome, texto))
        print(f"   ✓ {nome}")

    return resultado


# ── agente PydanticAI ─────────────────────────────────────────────────────────

def criar_agente() -> Agent:
    """
    Cria o agente com resultado tipado como AnaliseNPS.

    A diferença chave: result_type=AnaliseNPS faz com que:
    1. O LLM receba instruções para formatar o output como JSON
    2. O Pydantic valide e converta o JSON para um objeto Python
    3. resultado.data seja uma instância de AnaliseNPS — com tipos corretos

    Se o LLM retornar algo fora do schema, o PydanticAI tenta novamente
    automaticamente (até 3 tentativas por padrão) antes de levantar exceção.
    """
    return Agent(
        MODELO,
        output_type=AnaliseNPS,
        retries=3,
        system_prompt=(
            "Você é especialista em análise de satisfação de cursos de IA e tecnologia. "
            "Analisa feedbacks de pesquisas NPS com precisão e objetividade. "
            "Sempre cita trechos reais dos comentários dos alunos para sustentar "
            "suas conclusões. Responde sempre em português."
        )
    )


# ── execução principal ────────────────────────────────────────────────────────

def analisar_arquivo(agente: Agent, nome: str, texto: str) -> AnaliseNPS:
    """Analisa um arquivo de NPS e retorna objeto Python tipado."""

    prompt = f"""
Analise o seguinte relatório de NPS da aula "{nome}".

Extraia o score NPS (número entre 0-100 que aparece no documento),
leia todos os comentários textuais dos alunos, e preencha a análise completa.

RELATÓRIO NPS:
{texto[:6000]}
"""
    # run_sync() é a versão síncrona — use run() (async) em APIs e FastAPI
    resultado = agente.run_sync(prompt)

    # resultado.output é uma instância de AnaliseNPS — tipado e validado
    return resultado.output


def imprimir_analise(analise: AnaliseNPS) -> None:
    """Imprime a análise de forma legível. Note o acesso tipado aos campos."""
    print(f"\n{'─' * 60}")
    print(f"📄 {analise.nome_aula}")
    print(f"{'─' * 60}")

    # acesso direto ao campo — a IDE sabe que é float
    print(f"\n📊 NPS: {analise.nps_score:.0f}  |  {analise.sentimento_geral.upper()}  |  Engajamento: {analise.nivel_engajamento}")

    print(f"\n✅ ELOGIOS:")
    for i, elogio in enumerate(analise.elogios, 1):
        # analise.elogios é List[PontoFeedback] — o Pydantic garantiu isso
        freq = {"único": "1x", "alguns": "2-3x", "frequente": "4+x"}[elogio.frequencia]
        print(f"   {i}. {elogio.descricao} ({freq})")
        print(f'      → "{elogio.citacao}"')

    print(f"\n⚠️  CRÍTICAS:")
    for i, critica in enumerate(analise.criticas, 1):
        freq = {"único": "1x", "alguns": "2-3x", "frequente": "4+x"}[critica.frequencia]
        print(f"   {i}. {critica.descricao} ({freq})")
        print(f'      → "{critica.citacao}"')

    print(f"\n💡 INSIGHT: {analise.insight_inesperado}")
    print(f"\n🎯 AÇÃO URGENTE: {analise.recomendacao_urgente}")


def gerar_consolidado(agente: Agent, analises: list[AnaliseNPS]) -> None:
    """Gera um resumo consolidado a partir dos objetos tipados."""

    # Como os dados já são objetos Python tipados, podemos processar
    # sem depender de parsing de string — isso é o valor do PydanticAI

    nps_medio = sum(a.nps_score for a in analises) / len(analises)
    sentimentos = [a.sentimento_geral for a in analises]
    acoes = [a.recomendacao_urgente for a in analises]

    print(f"\n{'=' * 60}")
    print("  CONSOLIDADO — TODAS AS AULAS")
    print(f"{'=' * 60}")
    print(f"\n📊 NPS Médio Geral: {nps_medio:.1f}")
    print(f"\n📈 Sentimentos por aula:")
    for a in analises:
        emoji = {"muito positivo": "🟢", "positivo": "🟡", "neutro": "⚪",
                 "negativo": "🔴", "crítico": "🔴"}[a.sentimento_geral]
        print(f"   {emoji} {a.nome_aula[:40]:<40} NPS: {a.nps_score:.0f}  ({a.sentimento_geral})")

    print(f"\n🎯 Ações recomendadas por aula:")
    for a in analises:
        print(f"   • [{a.nome_aula[:30]}] {a.recomendacao_urgente}")


def main():
    print("=" * 60)
    print("  ANALISADOR DE NPS — PYDANTICAI")
    print("  Output tipado e validado automaticamente")
    print("=" * 60)

    arquivos = carregar_arquivos_nps()
    agente = criar_agente()

    analises: list[AnaliseNPS] = []

    for nome, texto in arquivos:
        print(f"\n⏳ Analisando: {nome}...")
        try:
            analise = analisar_arquivo(agente, nome, texto)
            imprimir_analise(analise)
            analises.append(analise)
        except Exception as e:
            print(f"   ❌ Erro ao analisar {nome}: {e}")

    if len(analises) > 1:
        gerar_consolidado(agente, analises)


if __name__ == "__main__":
    load_dotenv()
    main()
