"""
fallback_modelo.py
==================
Semana 08 — Nivelamento IA · Alura + Santander

Demo ao vivo: cinco abordagens para lidar com falha de API de LLM
    A) try/except manual                  ← Python puro, controle total
    B) with_fallbacks() do LangChain      ← mais elegante para LangChain
    C) cadeia encadeada Groq → Claude → GPT
    D) OpenRouter (fallback nativo)       ← sem código extra, um provider só
    E) PydanticAI FallbackModel           ← declarativo e tipado

Como simular falha durante a demo:
    - Abordagens A/B/C: export GROQ_API_KEY=chave_errada
    - Abordagem D: requer conta em openrouter.ai (OPENROUTER_API_KEY no .env)
    - Abordagem E: requer pydantic-ai instalado

Pré-requisitos:
    pip install langchain-groq langchain-anthropic langchain-openai python-dotenv
    pip install openai                    # para abordagem D (OpenRouter)
    pip install pydantic-ai              # para abordagem E
    pip install litellm                  # alternativa à E, usada pelo CrewAI
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────────────────────────
# ABORDAGEM A: try/except manual
# ────────────────────────────────────────────────────────────────────
def abordagem_a(pergunta: str) -> str:
    """Controle total do fluxo de erro. Mais verboso, mais explícito."""
    from langchain_groq import ChatGroq
    from langchain_anthropic import ChatAnthropic

    try:
        print("[A] Tentando Groq...")
        llm = ChatGroq(model="llama-3.1-8b-instant")
        resposta = llm.invoke(pergunta)
        print("[A] ✓ Groq respondeu")
        return resposta.content

    except Exception as e:
        print(f"[A] ✗ Groq falhou: {e}")
        print("[A] Tentando Claude como fallback...")
        try:
            llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
            resposta = llm.invoke(pergunta)
            print("[A] ✓ Claude respondeu")
            return resposta.content
        except Exception as e2:
            return f"[A] Todos os modelos falharam: {e2}"


# ────────────────────────────────────────────────────────────────────
# ABORDAGEM B: with_fallbacks() — nativo do LangChain
# ────────────────────────────────────────────────────────────────────
def abordagem_b(pergunta: str) -> str:
    """
    LangChain tenta cada modelo em sequência automaticamente.
    O código de uso é idêntico — a mágica é na configuração.
    """
    from langchain_groq import ChatGroq
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    # Definir modelos em ordem de preferência
    # api_key com fallback "invalid" garante que o objeto é criado mesmo sem a
    # variável de ambiente, e a falha ocorre no invoke() — onde with_fallbacks() age.
    groq   = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY", "invalid"))
    claude = ChatAnthropic(model="claude-haiku-4-5-20251001")
    gpt    = ChatOpenAI(model="gpt-4o-mini")

    # Criar cadeia com fallbacks automáticos
    llm = groq.with_fallbacks(
        [claude, gpt],
        exceptions_to_handle=(Exception,),  # qualquer erro aciona o próximo
    )

    print("[B] Invocando com fallback automático (Groq → Claude → GPT)...")
    resposta = llm.invoke(pergunta)
    print(f"[B] ✓ Respondido por: {resposta.response_metadata.get('model_name', 'desconhecido')}")
    return resposta.content


# ────────────────────────────────────────────────────────────────────
# ABORDAGEM C: cadeia encadeada com RunnableWithFallbacks em LCEL
# ────────────────────────────────────────────────────────────────────
def abordagem_c(pergunta: str) -> str:
    """
    Encadeamento explícito via LCEL.
    Útil quando você quer customizar o comportamento entre tentativas
    (ex: logar, alertar, mudar o prompt por modelo).
    """
    from langchain_groq import ChatGroq
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import RunnableWithFallbacks

    modelos = [
        ("Groq/Llama",  ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY", "invalid"))),
        ("Claude Haiku", ChatAnthropic(model="claude-haiku-4-5-20251001")),
        ("GPT-4o-mini",  ChatOpenAI(model="gpt-4o-mini")),
    ]

    for nome, llm in modelos:
        try:
            print(f"[C] Tentando {nome}...")
            resposta = llm.invoke(pergunta)
            print(f"[C] ✓ {nome} respondeu")
            # Em produção: logar qual modelo foi usado para monitoramento
            return resposta.content
        except Exception as e:
            print(f"[C] ✗ {nome} falhou: {type(e).__name__}")

    return "[C] Todos os modelos falharam."


# ────────────────────────────────────────────────────────────────────
# ABORDAGEM D: OpenRouter — fallback nativo de provider
# ────────────────────────────────────────────────────────────────────
def abordagem_d(pergunta: str) -> str:
    """
    OpenRouter como proxy unificado com fallback nativo.
    Mesma interface da OpenAI — só muda o base_url e o campo models[].
    Nenhum SDK extra além do openai.

    Pré-requisito: criar conta em openrouter.ai e obter OPENROUTER_API_KEY.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return "[D] OPENROUTER_API_KEY não encontrada no .env — pule esta abordagem."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print("[D] Chamando OpenRouter com fallback nativo...")
    resposta = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",    # obrigatório pelo SDK OpenAI
        extra_body={                                  # parâmetros extras do OpenRouter
            "models": [                               # lista de fallback do OpenRouter
                "meta-llama/llama-3.1-8b-instruct",
                "anthropic/claude-haiku-4-5",
                "openai/gpt-4o-mini",
            ],
            "route": "fallback",                      # tenta em sequência até ter resposta
        },
        messages=[{"role": "user", "content": pergunta}],
    )
    print("[D] ✓ OpenRouter respondeu (fallback gerenciado pelo provider)")
    return resposta.choices[0].message.content


# ────────────────────────────────────────────────────────────────────
# ABORDAGEM E: PydanticAI FallbackModel — declarativo e tipado
# ────────────────────────────────────────────────────────────────────
def abordagem_e(pergunta: str) -> str:
    """
    PydanticAI com FallbackModel: classe dedicada para fallback.
    Boa escolha para sistemas que já usam Pydantic e precisam de tipagem forte.

    Pré-requisito: pip install pydantic-ai
    """
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.fallback import FallbackModel
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.groq import GroqProvider
    except ImportError:
        return "[E] pydantic-ai não instalado — execute: pip install pydantic-ai"

    # FallbackModel tenta cada modelo em sequência automaticamente.
    # GroqProvider com api_key explícita evita erro no __init__ quando a
    # variável de ambiente está ausente — a falha ocorre no run_sync(), onde
    # o FallbackModel pode capturar e tentar o próximo modelo.
    model = FallbackModel(
        GroqModel("llama-3.1-8b-instant", provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY", "invalid"))),
        AnthropicModel("claude-haiku-4-5-20251001"),
    )

    agent = Agent(model=model)

    print("[E] Invocando via PydanticAI FallbackModel...")
    result = agent.run_sync(pergunta)
    print("[E] ✓ PydanticAI respondeu")
    return result.output


# ────────────────────────────────────────────────────────────────────
# BÔNUS: LiteLLM — o que o CrewAI usa por baixo
# ────────────────────────────────────────────────────────────────────
def abordagem_litellm(pergunta: str) -> str:
    """
    LiteLLM: interface unificada para 100+ modelos com fallback nativo.
    O CrewAI usa LiteLLM internamente — configurar fallback aqui equivale
    a configurar fallback em qualquer crew.

    Pré-requisito: pip install litellm
    """
    try:
        import litellm
    except ImportError:
        return "[LiteLLM] litellm não instalado — execute: pip install litellm"

    print("[LiteLLM] Invocando com fallback automático...")
    resposta = litellm.completion(
        model="groq/llama-3.1-8b-instant",
        fallbacks=[
            "anthropic/claude-haiku-4-5-20251001",
            "openai/gpt-4o-mini",
        ],
        messages=[{"role": "user", "content": pergunta}],
    )
    print("[LiteLLM] ✓ Respondeu")
    return resposta.choices[0].message.content


# ────────────────────────────────────────────────────────────────────
# DEMO AO VIVO
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PERGUNTA = "Em uma frase: o que é machine learning?"

    print("=" * 60)
    print("ABORDAGEM A — try/except manual")
    print("=" * 60)
    print(abordagem_a(PERGUNTA))

    print("\n" + "=" * 60)
    print("ABORDAGEM B — with_fallbacks() LangChain")
    print("=" * 60)
    print(abordagem_b(PERGUNTA))

    print("\n" + "=" * 60)
    print("ABORDAGEM C — cadeia encadeada manual")
    print("=" * 60)
    print(abordagem_c(PERGUNTA))

    print("\n" + "=" * 60)
    print("ABORDAGEM D — OpenRouter (fallback nativo de provider)")
    print("=" * 60)
    print(abordagem_d(PERGUNTA))

    print("\n" + "=" * 60)
    print("ABORDAGEM E — PydanticAI FallbackModel")
    print("=" * 60)
    print(abordagem_e(PERGUNTA))

    print("\n" + "=" * 60)
    print("BÔNUS — LiteLLM (base do CrewAI)")
    print("=" * 60)
    print(abordagem_litellm(PERGUNTA))

    # ── Dica para simular falha ao vivo ──────────────────────────
    print("\n💡 Para testar o fallback ao vivo:")
    print("   1. Exporte uma chave inválida: export GROQ_API_KEY=chave_errada")
    print("   2. Rode novamente — o modelo seguinte será chamado automaticamente")
    print("   3. Restaure: export GROQ_API_KEY=sua_chave_real")
    print("\n📋 Quando usar cada abordagem:")
    print("   A) Precisa de controle total e log personalizado")
    print("   B) Já usa LangChain — solução nativa e elegante")
    print("   C) Quer fallback sem gerenciar múltiplos SDKs")
    print("   D) Quer responsabilidade do provider, não do código")
    print("   E) Projeto Pydantic com tipagem forte")
