"""
kmeans_nps.py
=============
Semana 08 — Nivelamento IA · Alura + Santander

Demo ao vivo: segmentar comentários NPS em clusters com K-Means
e pedir para um LLM nomear e descrever cada grupo automaticamente.

Pré-requisitos (Colab ou local):
    pip install pandas scikit-learn langchain-groq python-dotenv
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── 1. Carregar dados ──────────────────────────────────────────────
CSV_PATH = "comentarios_nps_completo.csv"  # ajuste o caminho se necessário

df = pd.read_csv(CSV_PATH)
print(f"Total de comentários carregados: {len(df)}")
print(df[["aula", "nps", "comentario"]].head(3))
print()

# Limpar comentários vazios
comentarios = df["comentario"].fillna("").astype(str).tolist()
comentarios = [c for c in comentarios if len(c.strip()) > 5]
print(f"Comentários válidos para clustering: {len(comentarios)}")

# ── 2. Vetorizar com TF-IDF ────────────────────────────────────────
# TF-IDF transforma cada comentário num vetor numérico.
# max_features=150: usa as 150 palavras mais relevantes do corpus.
vectorizer = TfidfVectorizer(
    max_features=150,
    min_df=2,           # ignora palavras que aparecem em < 2 comentários
    strip_accents="unicode",
    lowercase=True,
)
X = vectorizer.fit_transform(comentarios)
print(f"\nMatriz TF-IDF: {X.shape[0]} comentários × {X.shape[1]} features")

# ── 3. Escolher o número de clusters (método do cotovelo) ──────────
# Opcional: descomente para ver o gráfico de inércia e silhouette

# import matplotlib.pyplot as plt
# inercias, silhouettes = [], []
# for k in range(2, 9):
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     km.fit(X)
#     inercias.append(km.inertia_)
#     silhouettes.append(silhouette_score(X, km.labels_))
# plt.plot(range(2, 9), silhouettes, marker="o")
# plt.title("Silhouette por k")
# plt.show()

N_CLUSTERS = 4  # valor razoável para ~150 comentários

# ── 4. Aplicar K-Means ─────────────────────────────────────────────
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df_validos = df[df["comentario"].fillna("").str.len() > 5].copy()
df_validos["cluster"] = kmeans.fit_predict(X)

print(f"\nDistribuição de comentários por cluster:")
print(df_validos["cluster"].value_counts().sort_index())

# ── 5. Ver amostras por cluster ────────────────────────────────────
print("\n── Amostra de comentários por cluster ──")
for c in range(N_CLUSTERS):
    subset = df_validos[df_validos["cluster"] == c]["comentario"].tolist()
    print(f"\nCluster {c} ({len(subset)} comentários):")
    for texto in subset[:3]:
        print(f"  • {texto[:90]}")

# ── 6. LLM nomeia e descreve cada cluster ─────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt_template = PromptTemplate.from_template(
    "Você é um especialista em análise de feedback de cursos online.\n"
    "Analise os comentários abaixo (todos do mesmo grupo) e responda:\n"
    "1. Um nome curto para este grupo (máx. 5 palavras)\n"
    "2. Um resumo do sentimento e tema principal (2 frases)\n"
    "3. Uma sugestão de ação para o professor\n\n"
    "Comentários:\n{comentarios}\n\n"
    "Responda em português, de forma direta e objetiva."
)

chain = prompt_template | llm

print("\n══ Análise dos clusters pelo LLM ══\n")
for c in range(N_CLUSTERS):
    subset = df_validos[df_validos["cluster"] == c]["comentario"].tolist()
    amostra = "\n".join(f"- {t}" for t in subset[:8])  # até 8 por cluster

    resposta = chain.invoke({"comentarios": amostra})
    print(f"── Cluster {c} ({len(subset)} comentários) ──")
    print(resposta.content)
    print()

# ── 7. Salvar resultado ────────────────────────────────────────────
output_path = "nps_clusters.csv"
df_validos[["aula", "nps", "comentario", "cluster"]].to_csv(output_path, index=False)
print(f"Resultado salvo em: {output_path}")
