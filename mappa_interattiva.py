import os
import re
import time
import pdfplumber
import matplotlib.pyplot as plt
import ast
from mistralai import Mistral, SDKError
import streamlit as st
import base64

# === CONFIGURAZIONE API ===
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
MODEL = st.secrets.get("MISTRAL_MODEL", "mistral-large-latest")

# === ESTRAZIONE E SUDDIVISIONE TESTO ===

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

def suddividi_testo(testo: str, max_chars: int = 15000) -> list[str]:
    parole = testo.split()
    blocchi, corrente, lunghezza = [], [], 0
    for parola in parole:
        if lunghezza + len(parola) + 1 > max_chars:
            blocchi.append(" ".join(corrente))
            corrente, lunghezza = [], 0
        corrente.append(parola)
        lunghezza += len(parola) + 1
    if corrente:
        blocchi.append(" ".join(corrente))
    return blocchi

# === GENERAZIONE STRUTTURA VIA LLM A BLOCCHI ===

def genera_struttura_per_blocco(block_text: str, central_node: str) -> dict:
    """
    Chiama il modello su un singolo blocco di testo
    e restituisce un dict {Branch: [sub1, sub2, ...], ...}
    """
    prompt = f"""
Leggi questo estratto di PDF e restituisci SOLO un dizionario Python
(nessun testo aggiuntivo né code fences) con i rami intorno a "{central_node}".

Formato atteso (in una singola riga):
{{
  "Branch A": ["Sub1", "Sub2"],
  "Branch B": ["Sub1", ...],
  ...
}}

Testo del blocco:
{block_text}
"""
    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = resp.choices[0].message.content.strip()
            return ast.literal_eval(txt)
        except (SDKError, ValueError, SyntaxError):
            time.sleep(2 ** attempt)
            continue
    return {}

def merge_structures(acc: dict[str, set[str]], part: dict[str, list[str]]):
    for b, subs in part.items():
        if b not in acc:
            acc[b] = set()
        for s in subs:
            acc[b].add(s)

# === DISEGNO MIND MAP ===

def draw_mind_map(central_node: str, branches: dict[str, set[str]]):
    primari = list(branches.keys())
    metà = len(primari) // 2
    left, right = primari[:metà], primari[metà:]

    def flatten(side):
        flat = []
        for b in side:
            flat.append((b, 0))
            for sub in sorted(branches[b]):
                flat.append((sub, 1, b))
        return flat

    flat_L, flat_R = flatten(left), flatten(right)

    def compute_pos(flat):
        n = len(flat)
        if n == 0:
            return {}
        ys = [0.9 - i*(0.8/(n-1)) for i in range(n)]
        return {item[0]: y for item, y in zip(flat, ys)}

    posL, posR = compute_pos(flat_L), compute_pos(flat_R)

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis("off")
    ax.text(0.5,0.5, central_node, fontsize=16, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="lightblue"))

    # linee ai rami
    for node,depth,*_ in flat_L:
        if depth==0:
            ax.plot([0.5,0.25],[0.5,posL[node]],"gray")
    for node,depth,*_ in flat_R:
        if depth==0:
            ax.plot([0.5,0.75],[0.5,posR[node]],"gray")

    # etichette e linee interne
    for node,depth,*rest in flat_L:
        x = 0.25 - 0.03*depth
        y = posL[node]
        txt = (f"- {node}") if depth else node
        ax.text(x,y,txt, fontsize=12 if depth==0 else 10,
                ha="center", va="center",
                bbox=dict(boxstyle="round",
                          fc="lavender" if depth==0 else "white"))
        if depth==1:
            parent = rest[0]
            ax.plot([0.25,0.25],[posL[parent],y],"gray")

    for node,depth,*rest in flat_R:
        x = 0.75 + 0.03*depth
        y = posR[node]
        txt = (f"- {node}") if depth else node
        ax.text(x,y,txt, fontsize=12 if depth==0 else 10,
                ha="center", va="center",
                bbox=dict(boxstyle="round",
                          fc="lavender" if depth==0 else "white"))
        if depth==1:
            parent = rest[0]
            ax.plot([0.75,0.75],[posR[parent],y],"gray")

    plt.tight_layout()
    st.pyplot(fig)

# === STREAMLIT APP ===

st.set_page_config(page_title="Mappa Concettuale a Blocchi", layout="wide")
st.title("Generatore Mappa Concettuale PDF — Output Diretto a Blocchi")

doc = st.file_uploader("Carica il PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")

if st.button("Genera e Mostra Mappa") and doc:
    # caricamento GIF
    gif_path = "img/Progetto video 1.gif"
    gif_ph = st.empty()
    if os.path.exists(gif_path):
        b64 = base64.b64encode(open(gif_path,"rb").read()).decode()
        gif_ph.markdown(f"<img src='data:image/gif;base64,{b64}' width=200/>", unsafe_allow_html=True)

    testo = estrai_testo_da_pdf(doc)
    blocchi = suddividi_testo(testo)

    status = st.empty()
    prog   = st.progress(0)
    accum: dict[str, set[str]] = {}

    total = len(blocchi)
    for idx, blk in enumerate(blocchi, start=1):
        pct = int((idx/total)*100)
        status.info(f"Generazione mappa... {pct}%")
        prog.progress(pct)

        part = genera_struttura_per_blocco(blk, central_node)
        merge_structures(accum, part)

    prog.empty()
    status.success("Mappa generata!")
    gif_ph.empty()

    if accum:
        draw_mind_map(central_node, accum)
    else:
        st.error("Non è stato possibile generare alcuna struttura.")
