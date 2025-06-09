import os
import re
import time
import json
import pdfplumber
import matplotlib.pyplot as plt
import streamlit as st
import base64
from mistralai import Mistral, SDKError
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# === CONFIGURAZIONE API ===
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
MODEL = st.secrets.get("MISTRAL_MODEL", "mistral-large-latest")

# === FUNZIONI COMUNI ===
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


def call_with_retries(prompt_args, max_retries: int = 5):
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(1)
            return client.chat.complete(**prompt_args)
        except SDKError as e:
            if e.status_code == 429 and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            else:
                raise

# === MAPPA GERARCHICA ===
def genera_struttura_per_blocco(block_text: str, central_node: str) -> dict:
    prompt = f"""
Leggi questo PDF per una descrizione dettagliata individuando i rami concettuali intorno al nodo centrale \"{central_node}\".
1. Individua tutti i concetti collegati al nodo centrale.
2. Fornisci una brevissima spiegazione.
3. Restituisci solo un JSON valido tra triple backticks con la forma:
```json
{{\"{central_node}\": {{\"Ramo A\": [\"Sottoramo1\", ...], ...}}}}
```
Nessun altro testo.
Estrai dai seguenti contenuti:
```
{block_text}
```"""
    resp = call_with_retries({'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}]})
    txt = resp.choices[0].message.content
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.S)
    if not m:
        m = re.search(r"```(?:python)?\s*(\{.*?\})\s*```", txt, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get(central_node, {})
        except json.JSONDecodeError:
            return {}
    return {}


def merge_structures(acc: dict[str, set[str]], part: dict[str, list[str]]):
    for ramo, subs in part.items():
        acc.setdefault(ramo, set()).update(subs)


def draw_mind_map(central_node: str, branches: dict[str, set[str]]):
    branches.pop(central_node, None)
    primari = list(branches.keys())
    meta = len(primari) // 2
    left, right = primari[:meta], primari[meta:]

    def flatten(side):
        flat = []
        for ramo in side:
            flat.append((ramo, 0))
            for sub in sorted(branches[ramo]):
                flat.append((sub, 1, ramo))
        return flat

    flat_L, flat_R = flatten(left), flatten(right)
    def compute_pos(flat):
        n = len(flat)
        ys = [0.9 - i * (0.8 / (n - 1)) for i in range(n)] if n > 1 else [0.5]
        return {item[0]: y for item, y in zip(flat, ys)}
    posL, posR = compute_pos(flat_L), compute_pos(flat_R)

    fig, ax = plt.subplots(figsize=(20, max(12, (len(flat_L) + len(flat_R) + 1) * 0.3)))
    ax.axis('off')
    total_nodes = len(flat_L) + len(flat_R) + 1
    main_fs = 16 if total_nodes <= 50 else max(8, 16 * 50 / total_nodes)
    sub_fs = main_fs * 0.8

    ax.text(0.5, 0.5, central_node, fontsize=main_fs, ha='center', va='center', bbox=dict(boxstyle='round', fc='lightblue'))
    for node, depth, *rest in flat_L:
        if depth == 0:
            ax.plot([0.5, 0.25], [0.5, posL[node]], 'gray')
    for node, depth, *rest in flat_R:
        if depth == 0:
            ax.plot([0.5, 0.75], [0.5, posR[node]], 'gray')
    for node, depth, *rest in flat_L:
        x = 0.25 - 0.03 * depth; y = posL[node]
        fs = main_fs if depth == 0 else sub_fs
        txt = node if depth == 0 else f"- {node}"
        ax.text(x, y, txt, fontsize=fs, ha='center', va='center', bbox=dict(boxstyle='round', fc='lavender' if depth == 0 else 'white'))
        if depth == 1:
            ax.plot([0.25, 0.25], [posL[rest[0]], y], 'gray')
    for node, depth, *rest in flat_R:
        x = 0.75 + 0.03 * depth; y = posR[node]
        fs = main_fs if depth == 0 else sub_fs
        txt = node if depth == 0 else f"- {node}"
        ax.text(x, y, txt, fontsize=fs, ha='center', va='center', bbox=dict(boxstyle='round', fc='lavender' if depth == 0 else 'white'))
        if depth == 1:
            ax.plot([0.75, 0.75], [posR[rest[0]], y], 'gray')
    plt.tight_layout()
    st.pyplot(fig)

# === PAGINAZIONE GERARCHICA ===
def paginate_and_draw(central_node: str, branches: dict[str, set[str]], per_page: int):
    keys = list(branches.keys())
    pages = [keys[i:i + per_page] for i in range(0, len(keys), per_page)]
    for page in pages:
        subset = {k: branches[k] for k in page}
        draw_mind_map(central_node, subset)

# === STREAMLIT APP ===
st.set_page_config(page_title="Generatore Mappa Concettuale PDF", layout="wide")
mode = st.sidebar.selectbox("Modalità", ["Gerarchica Paginata", "Circolare"])

doc = st.file_uploader("Carica PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")

if mode == "Gerarchica Paginata":
    per_page = st.number_input("Rami per immagine", min_value=1, max_value=50, value=10)
    if st.button("Genera Mappa"):  
        testo = estrai_testo_da_pdf(doc)
        blocchi = suddividi_testo(testo)
        accum: dict[str, set[str]] = {}
        prog = st.progress(0)
        total = len(blocchi)
        for idx, blk in enumerate(blocchi, 1):
            prog.progress(idx / total)
            part = genera_struttura_per_blocco(blk, central_node)
            merge_structures(accum, part)
        paginate_and_draw(central_node, accum, per_page)
else:
    if st.button("Genera Circolare"):
        testo = estrai_testo_da_pdf(doc)
        # circolare logic unchanged
        index_terms = estrai_indice(testo)
        mappa = genera_mappa_concettuale(testo, central_node, index_terms)
        st.json(mappa)
        soglia = st.number_input("Soglia occorrenze", min_value=1, value=1)
        if st.button("Visualizza Grafo"):
            html_file = crea_grafo_interattivo(mappa, central_node, soglia)
            content = open(html_file, 'r').read()
            components.html(content, height=600, scrolling=True)
            st.download_button("Scarica HTML", data=content, file_name="grafo.html", mime='text/html')
