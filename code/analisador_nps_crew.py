# =============================================================================
# ANÁLISE DE NPS COM CREWAI — EQUIPE DE AGENTES
# Imersão Alura + Santander 2026 — Semana 07
#
# pip install crewai litellm python-dotenv pdfplumber
# Configurar: arquivo .env com GROQ_API_KEY=gsk_...
# =============================================================================

import os
import glob
import pdfplumber
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
import crewai.llms.cache as _crew_cache

# Groq não tem adapter nativo no CrewAI, então cache_breakpoint vaza para a API
# e causa BadRequestError. Este patch transforma mark_cache_breakpoint em no-op.
_crew_cache.mark_cache_breakpoint = lambda msg: msg

# ── configuração ──────────────────────────────────────────────────────────────

PASTA_NPS = os.path.join(
    os.path.dirname(__file__),
    "..", "NPS"
)
MODELO = "llama-3.3-70b-versatile"


# ── leitura dos PDFs ──────────────────────────────────────────────────────────

def ler_pdf(caminho: str) -> str:
    with pdfplumber.open(caminho) as pdf:
        paginas = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(paginas)


def carregar_feedbacks() -> tuple[str, int]:
    """
    Lê todos os PDFs de NPS e retorna:
    - texto consolidado com todos os feedbacks
    - número de arquivos processados
    """
    padrao = os.path.join(PASTA_NPS, "*.pdf")
    arquivos = sorted(glob.glob(padrao))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum PDF encontrado em: {PASTA_NPS}")

    print(f"\n📂 {len(arquivos)} arquivo(s) de NPS encontrado(s):\n")
    partes = []
    for caminho in arquivos:
        nome = os.path.basename(caminho)
        texto = ler_pdf(caminho)
        partes.append(f"=== {nome} ===\n{texto}")
        print(f"   ✓ {nome}")

    return "\n\n".join(partes), len(arquivos)


# ── agentes ───────────────────────────────────────────────────────────────────
# Cada agente tem três propriedades fundamentais:
#   role      → o título/papel do especialista
#   goal      → o que ele está tentando alcançar
#   backstory → o contexto que molda o estilo de raciocínio
#
# O backstory parece decorativo mas afeta o output real —
# experimente trocá-lo e observe a diferença.

def criar_agentes(llm: LLM) -> tuple[Agent, Agent]:

    analista = Agent(
        role="Analista de Experiência do Aluno",
        goal=(
            "Extrair os padrões mais relevantes dos feedbacks de NPS, "
            "identificar o que está funcionando e o que está gerando insatisfação, "
            "incluindo nuances que uma leitura rápida deixaria passar."
        ),
        backstory=(
            "Você tem 10 anos de experiência analisando satisfação em cursos "
            "de tecnologia e bootcamps. Trabalhou com turmas de perfis muito "
            "diferentes — desde devs experientes até pessoas em transição de "
            "carreira. Sabe distinguir uma crítica pontual de um padrão "
            "estrutural, e encontra nos comentários mais curtos as pistas mais "
            "valiosas sobre o estado emocional da turma."
        ),
        llm=llm,
        verbose=True
    )

    consultor = Agent(
        role="Consultor Pedagógico Sênior",
        goal=(
            "Transformar os insights de NPS em um plano de ação concreto, "
            "priorizando o que tem maior impacto na retenção e no aprendizado, "
            "com métricas claras para medir o resultado de cada ação."
        ),
        backstory=(
            "Você é especialista em design instrucional para cursos de IA e "
            "tecnologia. Já ajudou mais de 20 programas a melhorar seu NPS em "
            "pelo menos 15 pontos. Cada recomendação sua vem com critérios de "
            "sucesso mensuráveis — você nunca sugere 'melhorar a comunicação' "
            "sem dizer exatamente o que medir e em quanto tempo."
        ),
        llm=llm,
        verbose=True
    )

    return analista, consultor


# ── tasks ─────────────────────────────────────────────────────────────────────
# Tasks definem o trabalho concreto.
# O output da primeira task é passado automaticamente para a segunda.

def criar_tasks(
    analista: Agent,
    consultor: Agent,
    feedbacks: str,
    n_aulas: int
) -> tuple[Task, Task]:

    task_analise = Task(
        description=f"""
Analise os feedbacks de NPS das {n_aulas} aulas abaixo.

{feedbacks}

Sua análise deve conter:
1. NPS médio estimado (calcule com base nos scores que encontrar)
2. Top 3 elogios mais recorrentes (com exemplos reais dos comentários)
3. Top 3 críticas mais recorrentes (com exemplos reais dos comentários)
4. Perfil emocional da turma (como as pessoas estão se sentindo no curso)
5. Um insight que pode ser facilmente ignorado mas é importante

Seja específico. Cite trechos reais dos comentários para sustentar cada ponto.
""",
        expected_output=(
            "Relatório de análise com NPS médio, elogios, críticas, "
            "perfil emocional da turma e insight não óbvio."
        ),
        agent=analista
    )

    task_plano = Task(
        description="""
Com base na análise de NPS que você recebeu, crie um plano de ação pedagógico.

O plano deve ter exatamente 3 ações prioritárias.
Para cada ação, inclua:
- O QUE fazer (descrição específica, não genérica)
- POR QUE priorizar esta ação (vínculo direto com os dados de NPS)
- COMO medir o sucesso (métrica concreta em até 30 dias)
- CUSTO DE IMPLEMENTAÇÃO (baixo / médio / alto — em termos de esforço)

No final, inclua uma ação bônus de impacto imediato que pode ser feita
ainda na próxima aula, sem preparação prévia.
""",
        expected_output=(
            "Plano de ação com 3 itens priorizados (o que, por que, como medir, custo) "
            "e 1 ação bônus de impacto imediato."
        ),
        agent=consultor
    )

    return task_analise, task_plano


# ── execução principal ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ANÁLISE DE NPS — CREWAI")
    print("  Analista + Consultor Pedagógico")
    print("=" * 60)

    # carrega os PDFs
    feedbacks, n_aulas = carregar_feedbacks()

    # inicializa o LLM (CrewAI 1.x usa LLM nativo, não ChatGroq do LangChain)
    load_dotenv()
    llm = LLM(
        model=f"groq/{MODELO}",
        temperature=0.3,
        max_retries=5,       # tenta até 5x automaticamente
        timeout=120,         # aguarda até 2 min por resposta
    )

    # cria os agentes
    analista, consultor = criar_agentes(llm)

    # cria as tasks passando os dados reais
    task_analise, task_plano = criar_tasks(analista, consultor, feedbacks, n_aulas)

    # monta a equipe
    # Process.sequential = o analista termina antes do consultor começar
    # O output do analista é passado automaticamente para o contexto do consultor
    crew = Crew(
        agents=[analista, consultor],
        tasks=[task_analise, task_plano],
        process=Process.sequential,
        verbose=True
    )

    print(f"\n🚀 Iniciando crew com {n_aulas} arquivo(s) de NPS...\n")

    # kickoff() executa toda a sequência e retorna o resultado final
    resultado = crew.kickoff()

    print("\n" + "=" * 60)
    print("  RESULTADO FINAL DO CREW")
    print("=" * 60)
    print(resultado.raw)


if __name__ == "__main__":
    main()
