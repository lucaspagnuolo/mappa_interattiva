import os
import re
import time
import json
import pdfplumber
import matplotlib.pyplot as plt
import streamlit as st
import base64
from mistralai import Mistral, SDKError

# === CONFIGURAZIONE API ===
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
MODEL = st.secrets.get("MISTRAL_MODEL", "mistral-large-latest")

# === ESTRAZIONE E SUDDIVISIONE TESTO ===

def estrai_testo_da_pdf(file) -> str:
    testo = []
    with pdfplumber.open(file) as pdf:
        total = len(pdf.pages)
        prog = st.progress(0)
        for i, pagina in enumerate(pdf.pages, 1):
            testo.append(pagina.extract_text() or "")
            prog.progress(i / total)
    prog.empty()
    return "\n".join(testo)


def suddividi_testo_con_overlap(testo: str, max_chars: int = 15000, overlap_chars: int = 500) -> list[str]:
    parole = testo.split()
    blocchi = []
    i = 0
    while i < len(parole):
        corrente = parole[i:i + max_chars]
        blocchi.append(" ".join(corrente))
        i += max_chars - overlap_chars
    return blocchi

# === GENERAZIONE STRUTTURA VIA LLM A BLOCCHI ===

def genera_struttura_per_blocco(block_text: str, central_node: str) -> dict:
    prompt = f"""
Leggi questo estratto di PDF e individua i rami concettuali intorno al nodo centrale "{central_node}".
**1.** Fornisci una brevissima spiegazione (una sola frase).
**2.** Poi restituisci **solo** un JSON valido **tra triple backticks**, con questa forma:

```json
{{
  "{central_node}": {{
    "Ramo A": ["Sottoramo1", "Sottoramo2", ...],
    "Ramo B": [...],
    ...
  }}
}}
```

Nessun altro testo fuori dal blocco JSON.

Estrai dai seguenti contenuti:
```
{block_text}
```
"""
    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model=MODEL,
                messages=[{"role":"user","content":prompt}],
            )
            txt = resp.choices[0].message.content
            m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.S|re.M)
            if not m:
                m = re.search(r"```(?:python)?\s*(\{.*?\})\s*```", txt, flags=re.S|re.M)
            if not m:
                continue
            js = m.group(1)
            data = json.loads(js)
            return data.get(central_node, {})
        except (SDKError, json.JSONDecodeError):
            time.sleep(2 ** attempt)
    return {}


def merge_structures(acc: dict[str, set[str]], part: dict[str, list[str]]):
    for ramo, subs in part.items():
        acc.setdefault(ramo, set()).update(subs)

# === DISEGNO MIND MAP ===

def draw_mind_map(central_node: str, branches: dict[str, set[str]]):
    branches.pop(central_node, None)
    primari = list(branches.keys())
    metà = len(primari)//2
    left, right = primari[:metà], primari[metà:]

    def flatten(side):
        flat = []
        for ramo in side:
            flat.append((ramo,0))
            for sub in sorted(branches[ramo]):
                flat.append((sub,1,ramo))
        return flat

    flat_L, flat_R = flatten(left), flatten(right)

    def compute_pos(flat):
        n = len(flat)
        if n==0:
            return {}
        ys = [0.9 - i*(0.8/(n-1)) for i in range(n)]
        return {item[0]:y for item,y in zip(flat,ys)}

    posL, posR = compute_pos(flat_L), compute_pos(flat_R)

    fig, ax = plt.subplots(figsize=(20,12))
    ax.axis("off")
    ax.text(0.5,0.5,central_node,fontsize=16,ha="center",va="center",
            bbox=dict(boxstyle="round",fc="lightblue"))

    for node,depth,*_ in flat_L:
        if depth==0:
            ax.plot([0.5,0.25],[0.5,posL[node]],"gray")
    for node,depth,*_ in flat_R:
        if depth==0:
            ax.plot([0.5,0.75],[0.5,posR[node]],"gray")

    for node,depth,*rest in flat_L:
        x = 0.25 - 0.03*depth
        y = posL[node]
        txt = node if depth==0 else f"- {node}"
        ax.text(x,y,txt,fontsize=12 if depth==0 else 10,ha="center",va="center",
                bbox=dict(boxstyle="round",fc="lavender" if depth==0 else "white"))
        if depth==1:
            parent=rest[0]; ax.plot([0.25,0.25],[posL[parent],y],"gray")

    for node,depth,*rest in flat_R:
        x = 0.75 + 0.03*depth
        y = posR[node]
        txt = node if depth==0 else f"- {node}"
        ax.text(x,y,txt,fontsize=12 if depth==0 else 10,ha="center",va="center",
                bbox=dict(boxstyle="round",fc="lavender" if depth==0 else "white"))
        if depth==1:
            parent=rest[0]; ax.plot([0.75,0.75],[posR[parent],y],"gray")

    plt.tight_layout()
    st.pyplot(fig)

# === STREAMLIT APP ===

st.set_page_config(page_title="Mappa Concettuale a Blocchi", layout="wide")
st.title("Generatore Mappa Concettuale PDF — Output Diretto a Blocchi")

doc = st.file_uploader("Carica il PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")

if st.button("Genera e Mostra Mappa") and doc:
    gif_ph = st.empty()
    gif_path = "img/Progetto video 1.gif"
    if os.path.exists(gif_path):
        b64 = base64.b64encode(open(gif_path,"rb").read()).decode()
        gif_ph.markdown(f"<img src='data:image/gif;base64,{b64}' width=200/>", unsafe_allow_html=True)

    testo = estrai_testo_da_pdf(doc)
    blocchi = suddividi_testo_con_overlap(testo, max_chars=15000, overlap_chars=500)

    status, prog = st.empty(), st.progress(0)
    accum: dict[str, set[str]] = {}
    total = len(blocchi)

    for idx, blk in enumerate(blocchi, 1):
        pct = int(idx/total*100)
        status.info(f"Generazione mappa... {pct}%")
        prog.progress(pct)

        # sempre genera struttura
        part = genera_struttura_per_blocco(blk, central_node)
        # fallback se vuoto
        if not part:
            st.warning(f"Blocco {idx}: fallback automatico")
            part = {"[automatic]": [central_node]}

        st.markdown(f"**Blocco {idx}**")
        st.code(part, language="python")
        merge_structures(accum, part)

    prog.empty(); status.success("Mappa generata!"); gif_ph.empty()

    if accum:
        draw_mind_map(central_node, accum)
    else:
        st.error("Nessuna struttura trovata.")
