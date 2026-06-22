"""
atendimento_cliente.py
======================
Semana 08 — Nivelamento IA · Alura + Santander

Demo ao vivo: chatbot de atendimento ao cliente com LangChain
usando memória de conversa. O bot lembra o que foi dito antes
e responde dentro do contexto da empresa (EduTech Pro).

Como rodar:
    python atendimento_cliente.py

Pré-requisitos:
    pip install langchain-groq langchain-core python-dotenv
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── Base de conhecimento da empresa ───────────────────────────────
# Em produção: substituir por conteúdo do site, FAQ em PDF,
# ou recuperação vetorial (RAG) para bases maiores.
BASE_CONHECIMENTO = """
EMPRESA: EduTech Pro — Plataforma de Cursos Online

PLANOS E PREÇOS:
- Plano Básico: R$49/mês — acesso a cursos fundamentais, certificado digital
- Plano Pro: R$99/mês — todos os cursos, projetos guiados, mentoria mensal
- Plano Anual Pro: R$799/ano — economia de R$389 vs mensal

CANCELAMENTO E REEMBOLSO:
- Cancelamento a qualquer momento pelo painel do aluno
- Reembolso total em até 7 dias corridos após a compra (garantia incondicional)
- Após 7 dias: sem reembolso, mas acesso até o fim do período pago

SUPORTE:
- Horário: segunda a sexta, 9h às 18h (horário de Brasília)
- Canal: chat no site ou email suporte@edutechpro.com.br
- Tempo médio de resposta: até 4 horas úteis

CURSOS DISPONÍVEIS:
- Trilha Dados: Python, SQL, Machine Learning, Power BI
- Trilha Web: HTML/CSS, JavaScript, React, Node.js
- Trilha IA: Prompt Engineering, LangChain, Agentes de IA, Computer Vision
"""

# ── Configurar o chatbot ───────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é o assistente virtual da EduTech Pro, uma plataforma de cursos online.\n"
        "Responda dúvidas dos alunos com base nas informações abaixo.\n"
        "Se não souber a resposta, diga que vai verificar e peça o email do aluno.\n"
        "Seja amigável, objetivo e use no máximo 3 frases por resposta.\n\n"
        "BASE DE CONHECIMENTO:\n{base}"
    ),
    MessagesPlaceholder("historico"),
    ("human", "{pergunta}"),
])

chain = prompt | llm

# ── Memória da conversa ────────────────────────────────────────────
historico: list = []

def atender(pergunta: str) -> str:
    """
    Recebe uma pergunta do usuário, consulta o LLM com o histórico
    completo da conversa e retorna a resposta.
    """
    resposta = chain.invoke({
        "base": BASE_CONHECIMENTO,
        "historico": historico,
        "pergunta": pergunta,
    })

    # Adicionar ao histórico para a próxima mensagem
    historico.append(HumanMessage(content=pergunta))
    historico.append(AIMessage(content=resposta.content))

    return resposta.content


def mostrar_historico():
    """Exibe o histórico da conversa de forma legível."""
    print("\n── Histórico completo ──")
    for msg in historico:
        papel = "Aluno" if isinstance(msg, HumanMessage) else "Bot"
        print(f"{papel}: {msg.content}")
    print()


# ── Demo ao vivo ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Chatbot EduTech Pro — Demo ao vivo")
    print("=" * 60)
    print("(Digite 'sair' para encerrar, 'historico' para ver o log)\n")

    # Sequência de perguntas sugerida para a demo:
    perguntas_demo = [
        "Qual o preço do plano Pro?",
        "E se eu quiser cancelar depois?",       # testa memória: "E se" refere ao plano Pro
        "Quanto tempo tenho de reembolso?",      # testa contexto: sem repetir "cancelar"
        "Vocês têm cursos de IA?",
    ]

    modo_demo = input("Rodar sequência de perguntas da demo? (s/n): ").strip().lower()

    if modo_demo == "s":
        for pergunta in perguntas_demo:
            print(f"\nAluno: {pergunta}")
            resposta = atender(pergunta)
            print(f"Bot: {resposta}")
        mostrar_historico()
    else:
        # Modo interativo
        while True:
            entrada = input("\nAluno: ").strip()
            if not entrada:
                continue
            if entrada.lower() == "sair":
                print("Encerrando. Até mais!")
                break
            if entrada.lower() == "historico":
                mostrar_historico()
                continue
            print(f"Bot: {atender(entrada)}")

    print("\n💡 Principais conceitos demonstrados:")
    print("   • MessagesPlaceholder: injeta o histórico no prompt automaticamente")
    print("   • HumanMessage / AIMessage: tipagem explícita das mensagens")
    print("   • A lista 'historico' cresce a cada turno — o modelo vê tudo")
    print("   • Em produção: limitar o histórico (ex: últimas 20 mensagens)")