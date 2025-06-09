import os
import re
import time
import pdfplumber
import matplotlib.pyplot as plt
from mistralai import Mistral, SDKError
import streamlit as st
import base64
from PIL import Image
import ast

# === CONFIGURAZIONE API ===
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
MODEL = st.secrets.get("MISTRAL_MODEL", "mistral-large-latest")

# === ESTRAZIONE TESTO ===
def estrai_testo_da_pdf(file) -> str:
    testo = []
    with pdfplumber.open(file) as pdf:
        total = len(pdf.pages)
        progress = st.progress(0)
        for i, pagina in enumerate(pdf.pages, 1):
            testo.append(pagina.extract_text() or "")
            progress.progress(i / total)
    progress.empty()
    return "\n".join(testo)

# === CHIAMATA MODELLO PER STRUTTURA AD ALBERO ===
def genera_struttura_concettuale(testo: str, central_node: str) -> dict:
    """
    Chiede al LLM di restituire una struttura ad albero Python literal,
    es. {'Supporto IT': ['Gestione ticket', ...], ...}
    """
    prompt = f"""
    Leggi questo testo estratto da un PDF e costruisci una mappa concettuale
    con nodo centrale "{central_node}". Rispondi **SOLO** con un dizionario
    Python in una singola riga, della forma:

      {{
        "{central_node}": {{
          "Branch A": ["Sub1", "Sub2", ...],
          "Branch B": ["Sub1", ...],
          ...
        }}
      }}

    Dove:
    - Le chiavi di primo livello sono i rami direttamente collegati al nodo centrale.
    - Le liste contengono i sottorami di secondo livello.
    Non inserire altro testo, né code fences.
    
    Testo:
    {testo}
    """
    # una singola chiamata
    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            # parsing sicuro
            tree = ast.literal_eval(text)
            if isinstance(tree, dict):
                return tree[central_node]
        except SDKError as e:
            time.sleep(2 ** attempt)
            continue
        except Exception:
            break
    st.error("Errore nel parsing della struttura dal modello.")
    return {}

# === DISEGNO CON MATPLOTLIB ===
def draw_mind_map(central_node: str, branches: dict[str, list[str]]):
    """
    branches: dict di {branch: [sub1, sub2, ...], ...}
    disegna con distribuzione automatica per evitare sovrapposizioni.
    """
    # 1) divido i rami in due lati
    primari = list(branches.keys())
    meta = len(primari) // 2
    left = primari[:meta]
    right = primari[meta:]

    # 2) flatten con profondità
    def flatten(side):
        flat = []
        for b in side:
            flat.append((b, 0))
            for sub in branches.get(b, []):
                flat.append((sub, 1, b))
        return flat

    flat_L = flatten(left)
    flat_R = flatten(right)

    # 3) calcola y
    def pos(flat):
        n = len(flat)
        if n == 0:
            return {}
        ys = [0.9 - i * (0.8 / (n - 1)) for i in range(n)]
        return { item[0]: y for item, y in zip(flat, ys) }

    posL = pos(flat_L)
    posR = pos(flat_R)

    # 4) disegno
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis("off")

    # nodo centrale
    ax.text(0.5, 0.5, central_node,
            fontsize=16, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="lightblue"))

    # linee ai rami
    for node,depth,*rest in flat_L:
        if depth == 0:
            ax.plot([0.5, 0.25], [0.5, posL[node]], "gray")
    for node,depth,*rest in flat_R:
        if depth == 0:
            ax.plot([0.5, 0.75], [0.5, posR[node]], "gray")

    # etichette e profondità
    for node, depth, *rest in flat_L:
        x = 0.25 - 0.03 * depth
        y = posL[node]
        txt = (f"- {node}") if depth else node
        ax.text(x, y, txt,
                fontsize=12 if depth == 0 else 10,
                ha="center", va="center",
                bbox=dict(boxstyle="round",
                          fc="lavender" if depth == 0 else "white"))
        # linea ramo→sottoramo
        if depth == 1:
            parent = rest[0]
            ax.plot([0.25, 0.25], [posL[parent], y], "gray")

    for node, depth, *rest in flat_R:
        x = 0.75 + 0.03 * depth
        y = posR[node]
        txt = (f"- {node}") if depth else node
        ax.text(x, y, txt,
                fontsize=12 if depth == 0 else 10,
                ha="center", va="center",
                bbox=dict(boxstyle="round",
                          fc="lavender" if depth == 0 else "white"))
        if depth == 1:
            parent = rest[0]
            ax.plot([0.75, 0.75], [posR[parent], y], "gray")

    plt.tight_layout()
    st.pyplot(fig)

# === STREAMLIT APP ===
st.set_page_config(page_title="Mappa Concettuale Diretta", layout="wide")
st.title("Generatore Mappa Concettuale PDF — Output Diretto")

# caricamento
doc = st.file_uploader("Carica il PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")
gif_placeholder = st.empty()
gif_path = "img/Progetto video 1.gif"

if st.button("Genera e Mostra Mappa") and doc:
    # GIF di caricamento
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        gif_placeholder.markdown(
            f"<img src='data:image/gif;base64,{b64}' width=200/>",
            unsafe_allow_html=True
        )
    testo = estrai_testo_da_pdf(doc)
    # genera albero
    tree = genera_struttura_concettuale(testo, central_node)
    gif_placeholder.empty()

    if tree:
        # disegna
        draw_mind_map(central_node, tree)
    else:
        st.error("Non è stato possibile creare la mappa.")
