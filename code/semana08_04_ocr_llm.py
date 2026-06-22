"""
ocr_llm.py
==========
Semana 08 — Nivelamento IA · Alura + Santander

Demo ao vivo: dois pipelines de OCR + LLM para estruturar texto de imagens

ABORDAGEM A — Tesseract OCR (local, gratuito)
    imagem → pytesseract → texto bruto → LLM → JSON estruturado

ABORDAGEM B — Claude Vision API (cloud, sem instalação extra)
    imagem → base64 → API Anthropic → JSON estruturado direto

Caso de uso: digitalizar nota fiscal e extrair data, valor e itens.

Pré-requisitos:
    # Tesseract (apenas para Abordagem A):
    # Ubuntu/Colab: !apt install tesseract-ocr tesseract-ocr-por -y
    # macOS:        brew install tesseract tesseract-lang
    pip install pytesseract pillow langchain-groq langchain-core anthropic python-dotenv
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────────────────────────
# ABORDAGEM A: Tesseract OCR + LLM (local e gratuito)
# ────────────────────────────────────────────────────────────────────
def ocr_tesseract_llm(caminho_imagem: str) -> dict:
    """
    Passo 1: Tesseract extrai texto bruto da imagem (sem entender contexto).
    Passo 2: LLM interpreta o texto bruto e devolve JSON estruturado.
    """
    import pytesseract
    from PIL import Image
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate

    # Passo 1: OCR — extrair texto bruto
    print("[A] Extraindo texto com Tesseract...")
    imagem = Image.open(caminho_imagem)
    texto_bruto = pytesseract.image_to_string(imagem, lang="por")
    print(f"[A] Texto extraído ({len(texto_bruto)} chars):\n{texto_bruto[:300]}...\n")

    # Passo 2: LLM estrutura o texto em JSON
    print("[A] Enviando para LLM estruturar...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = PromptTemplate.from_template(
        "Analise o texto abaixo extraído de uma nota fiscal por OCR.\n"
        "Extraia as seguintes informações em formato JSON:\n"
        "- data_emissao\n"
        "- cnpj_emitente\n"
        "- valor_total\n"
        "- itens (lista com nome e valor de cada item)\n\n"
        "Se algum campo não for encontrado, use null.\n"
        "Retorne APENAS o JSON, sem explicações.\n\n"
        "Texto OCR:\n{texto}"
    )

    chain = prompt | llm
    resposta = chain.invoke({"texto": texto_bruto})

    print(f"[A] Resultado:\n{resposta.content}")
    return {"metodo": "tesseract", "resultado": resposta.content}


# ────────────────────────────────────────────────────────────────────
# ABORDAGEM B: Claude Vision API (cloud, alta precisão)
# ────────────────────────────────────────────────────────────────────
def ocr_claude_vision(caminho_imagem: str) -> dict:
    """
    Envia a imagem diretamente para o Claude via API.
    O modelo vê a imagem e extrai o JSON sem precisar de OCR separado.
    Vantagem: funciona mesmo em imagens tortas, com ruído ou baixa resolução.
    """
    import anthropic

    print("[B] Enviando imagem para Claude Vision...")

    # Converter imagem para base64
    with open(caminho_imagem, "rb") as f:
        dados_imagem = base64.standard_b64encode(f.read()).decode("utf-8")

    # Detectar tipo da imagem
    extensao = Path(caminho_imagem).suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    media_type = media_types.get(extensao, "image/jpeg")

    client = anthropic.Anthropic()

    mensagem = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": dados_imagem,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Esta é uma nota fiscal. Extraia em JSON:\n"
                            "- data_emissao\n"
                            "- cnpj_emitente\n"
                            "- valor_total\n"
                            "- itens (lista com nome e valor)\n\n"
                            "Se algum campo não estiver visível, use null.\n"
                            "Retorne APENAS o JSON, sem explicações."
                        ),
                    },
                ],
            }
        ],
    )

    resultado = mensagem.content[0].text
    print(f"[B] Resultado:\n{resultado}")
    return {"metodo": "claude_vision", "resultado": resultado}


# ────────────────────────────────────────────────────────────────────
# CRIAR IMAGEM DE TESTE (se não tiver uma nota fiscal real)
# ────────────────────────────────────────────────────────────────────
def criar_nota_fiscal_simulada(caminho: str = "nota_fiscal_teste.png"):
    """Gera uma imagem simples de nota fiscal para teste."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow não encontrado. pip install pillow")
        return None

    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    linhas = [
        "NOTA FISCAL ELETRONICA",
        "=" * 40,
        "EMITENTE: Papelaria São Paulo Ltda",
        "CNPJ: 12.345.678/0001-90",
        "DATA: 21/06/2026",
        "-" * 40,
        "ITENS:",
        "  Caderno universitário  x2  R$ 38,00",
        "  Caneta esferográfica   x5  R$ 12,50",
        "  Post-it colorido       x1  R$  9,90",
        "-" * 40,
        "TOTAL: R$ 60,40",
        "FORMA: Cartão de crédito",
    ]

    y = 20
    for linha in linhas:
        draw.text((20, y), linha, fill=(0, 0, 0))
        y += 28

    img.save(caminho)
    print(f"Imagem de teste criada: {caminho}")
    return caminho


# ────────────────────────────────────────────────────────────────────
# DEMO AO VIVO
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("OCR + LLM — Demo ao vivo")
    print("=" * 60)

    # Usar imagem fornecida ou criar uma de teste
    caminho = "nota-fiscal-teste.jpeg"
    if not Path(caminho).exists():
        print(f"\nArquivo '{caminho}' não encontrado.")
        print("Criando nota fiscal simulada para teste...")
        caminho = criar_nota_fiscal_simulada()
        if not caminho:
            print("Erro ao criar imagem de teste. Forneça uma imagem PNG ou JPG.")
            exit(1)

    print(f"\nUsando imagem: {caminho}\n")

    # Escolher abordagem
    print("Qual abordagem testar?")
    print("  A) Tesseract OCR + Groq/LLM (local)")
    print("  B) Claude Vision API (cloud)")
    print("  C) Ambas (comparação)")
    escolha = input("\nEscolha (A/B/C): ").strip().upper()

    if escolha in ("A", "C"):
        print("\n── ABORDAGEM A: Tesseract ──")
        try:
            ocr_tesseract_llm(caminho)
        except ImportError:
            print("pytesseract não instalado. Execute: pip install pytesseract")
            print("E instale o Tesseract: apt install tesseract-ocr tesseract-ocr-por")

    if escolha in ("B", "C"):
        print("\n── ABORDAGEM B: Claude Vision ──")
        ocr_claude_vision(caminho)

    print("\n💡 Caso de uso real:")
    print("   • Digitalizar pilha de notas fiscais físicas")
    print("   • Extrair automaticamente data, CNPJ e valor")
    print("   • Classificar por categoria de despesa")
    print("   • Integrar com planilha ou sistema contábil")
