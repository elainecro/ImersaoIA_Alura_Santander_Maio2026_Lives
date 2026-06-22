"""
visao_mobilenet.py
==================
Semana 08 — Nivelamento IA · Alura + Santander

Demo ao vivo: reconhecimento de objetos com MobileNetV2 (Transfer Learning)
Treinado com 1,2 milhão de imagens ImageNet — você usa pronto, sem treinar nada.

Funciona em:
    • Colab (recomendado para a demo ao vivo — upload de foto interativo)
    • Local (forneça o caminho da imagem via argumento)

Pré-requisitos:
    pip install tensorflow pillow numpy

No Colab: TensorFlow já vem instalado.
"""

import os
import sys
import numpy as np
from pathlib import Path


# ────────────────────────────────────────────────────────────────────
# UTILITÁRIO: detectar ambiente
# ────────────────────────────────────────────────────────────────────
def esta_no_colab() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False


# ────────────────────────────────────────────────────────────────────
# PASSO 1: Obter imagem (Colab ou local)
# ────────────────────────────────────────────────────────────────────
def obter_imagem() -> str:
    """Retorna o caminho da imagem, via upload no Colab ou argumento local."""

    if esta_no_colab():
        from google.colab import files
        print("📤 Envie uma foto para identificar (qualquer objeto funciona!)")
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("Nenhum arquivo enviado.")
        caminho = list(uploaded.keys())[0]
        print(f"✓ Arquivo recebido: {caminho}")
        return caminho

    # Modo local: argumento ou padrão
    if len(sys.argv) > 1:
        caminho = sys.argv[1]
    else:
        caminho = "objetos-01.jpeg"  # coloque sua imagem aqui

    if not Path(caminho).exists():
        print(f"Arquivo não encontrado: {caminho}")
        print("Criando imagem de teste sintética...")
        caminho = criar_imagem_teste()

    return caminho


# ────────────────────────────────────────────────────────────────────
# PASSO 2: Carregar o modelo MobileNetV2
# ────────────────────────────────────────────────────────────────────
def carregar_modelo():
    """
    Carrega o MobileNetV2 pré-treinado com ImageNet.
    Na primeira execução: download automático ~15MB.
    """
    from tensorflow.keras.applications import MobileNetV2

    print("\n🧠 Carregando MobileNetV2 (pré-treinado com ImageNet)...")
    modelo = MobileNetV2(weights="imagenet")
    print(f"✓ Modelo carregado — {modelo.count_params():,} parâmetros")
    print(f"  Entrada esperada: {modelo.input_shape}")
    return modelo


# ────────────────────────────────────────────────────────────────────
# PASSO 3: Pré-processar a imagem
# ────────────────────────────────────────────────────────────────────
def preprocessar(caminho: str) -> np.ndarray:
    """
    MobileNetV2 espera imagens 224×224 pixels, normalizadas entre -1 e 1.
    """
    from PIL import Image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    imagem = Image.open(caminho).convert("RGB")
    imagem_redimensionada = imagem.resize((224, 224))

    array = np.array(imagem_redimensionada)        # (224, 224, 3)
    array = np.expand_dims(array, axis=0)          # (1, 224, 224, 3) — batch de 1
    array_processado = preprocess_input(array)     # normaliza para [-1, 1]

    return array_processado


# ────────────────────────────────────────────────────────────────────
# PASSO 4: Classificar e exibir resultados
# ────────────────────────────────────────────────────────────────────
def classificar(modelo, array_processado: np.ndarray, top_k: int = 5):
    """
    Roda a imagem pelo modelo e decodifica as top_k previsões.
    """
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

    print("\n🔍 Classificando imagem...")
    predicoes = modelo.predict(array_processado, verbose=0)

    resultados = decode_predictions(predicoes, top=top_k)[0]

    print(f"\n── Top {top_k} objetos identificados ──")
    for i, (codigo, nome, confianca) in enumerate(resultados, 1):
        barra = "█" * int(confianca * 30)
        print(f"  {i}. {nome:<25} {confianca*100:5.1f}% {barra}")

    nome_principal = resultados[0][1].replace("_", " ")
    confianca_principal = resultados[0][2]
    print(f"\n✅ Objeto identificado: {nome_principal} ({confianca_principal*100:.1f}% de confiança)")
    return resultados


# ────────────────────────────────────────────────────────────────────
# BÔNUS: Explicar Transfer Learning para a turma
# ────────────────────────────────────────────────────────────────────
def explicar_transfer_learning():
    print("\n" + "=" * 60)
    print("📚 Por que o MobileNetV2 funciona tão bem?")
    print("=" * 60)
    print("""
Transfer Learning em 4 pontos:

1. TREINAMENTO ORIGINAL
   • 1.2 milhão de imagens, 1000 categorias, semanas de GPU
   • Custo estimado: dezenas de milhares de dólares

2. O QUE O MODELO APRENDEU
   • Camadas iniciais: detectar bordas, cores, texturas
   • Camadas médias: formas, partes de objetos (olho, roda, folha)
   • Camadas finais: conceitos completos (cachorro, carro, cadeira)

3. VOCÊ HERDA TUDO ISSO
   • model = MobileNetV2(weights='imagenet')   ← duas palavras
   • Nenhum dado de treino necessário
   • Inferência em < 1 segundo no Colab

4. FINE-TUNING (próximo nível)
   • Congele as camadas iniciais (já treinadas)
   • Retreine só as últimas com SEU dataset
   • 100-500 imagens já são suficientes para muitos casos
""")


# ────────────────────────────────────────────────────────────────────
# AUXILIAR: criar imagem de teste (quando não tem foto disponível)
# ────────────────────────────────────────────────────────────────────
def criar_imagem_teste(caminho: str = "objeto_teste.jpg") -> str:
    """Cria uma imagem simples com formas geométricas para teste."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 400), color=(200, 220, 255))
    draw = ImageDraw.Draw(img)

    # Simular uma forma que pode ser reconhecida
    draw.ellipse([100, 100, 300, 300], fill=(210, 180, 140), outline=(139, 69, 19), width=4)
    draw.rectangle([140, 200, 260, 280], fill=(101, 67, 33))
    draw.text((120, 340), "Objeto de teste", fill=(50, 50, 50))

    img.save(caminho)
    print(f"Imagem de teste criada: {caminho}")
    return caminho


# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MobileNetV2 — Reconhecimento de objetos ao vivo")
    print("=" * 60)

    if esta_no_colab():
        # No Colab: fluxo original (upload único)
        caminho = obter_imagem()
        modelo = carregar_modelo()
        array = preprocessar(caminho)
        print(f"✓ Imagem processada: {array.shape}")
        classificar(modelo, array, top_k=5)
    else:
        # Local: encontra todas as imagens que começam com "objetos-"
        pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
        extensoes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        imagens = sorted(
            p for p in pasta.glob("objetos-*")
            if p.suffix.lower() in extensoes
        )

        if not imagens:
            print(f"Nenhuma imagem 'objetos-*' encontrada em: {pasta.resolve()}")
            sys.exit(1)

        print(f"\n{len(imagens)} imagem(ns) encontrada(s):")
        for p in imagens:
            print(f"  • {p.name}")

        # Carrega o modelo uma única vez para todas as imagens
        modelo = carregar_modelo()

        for caminho in imagens:
            print("\n" + "=" * 60)
            print(f"Imagem: {caminho.name}")
            print("=" * 60)
            array = preprocessar(str(caminho))
            classificar(modelo, array, top_k=5)

    # Explicar Transfer Learning (opcional — para a aula)
    explicar = input("\nMostrar explicação de Transfer Learning? (s/n): ").strip().lower()
    if explicar == "s":
        explicar_transfer_learning()

    print("\n🎯 Demo concluída!")
    print("   Repositório de sugestões para fine-tuning:")
    print("   • Kaggle Datasets: kaggle.com/datasets")
    print("   • TensorFlow Datasets: tensorflow.org/datasets")
