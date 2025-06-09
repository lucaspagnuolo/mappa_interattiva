import os
import json
import re
import time
import pdfplumber
import networkx as nx
import matplotlib.pyplot as plt
from mistralai import Mistral, SDKError
import streamlit as st
import streamlit.components.v1 as components
import base64
from PIL import Image

# === CONFIGURAZIONE API ===
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
MODEL = st.secrets.get("MISTRAL_MODEL", "mistral-large-latest")

# === FUNZIONI DI BACKEND ===

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

def estrai_indice(testo: str) -> list[str]:
    righe = testo.splitlines()
    try:
        start = next(i for i, r in enumerate(righe)
                     if re.match(r'^(Indice|Sommario)\b', r, re.IGNORECASE))
    except StopIteration:
        return []
    termini = []
    for r in righe[start + 1:]:
        if not r.strip():
            break
        m = re.match(r'^(?P<termine>.+?)\s+\.{2,}\s*\d+|\s+\d+$', r)
        if m:
            termini.append(m.group('termine').strip())
        else:
            parti = r.rsplit(' ', 1)
            if len(parti) == 2 and parti[1].isdigit():
                termini.append(parti[0].strip())
    return termini

def filtra_paragrafi_sottoparagrafi(index_terms: list[str]) -> list[str]:
    pattern = re.compile(r'^\d+(?:\.\d+)*\s+[A-ZÀ-ÖØ-Ý]')
    return [t for t in index_terms if pattern.match(t)]

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

def call_with_retries(prompt_args, max_retries=5):
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

def genera_mappa_concettuale(testo: str, central_node: str, index_terms: list[str] = None) -> dict:
    blocchi = suddividi_testo(testo)
    ris = []
    status_text = st.empty()
    progress = st.progress(0)
    totale_blocchi = len(blocchi)

    for idx, b in enumerate(blocchi, 1):
        percentuale = int(((idx - 1) / totale_blocchi) * 100)
        status_text.info(f"Generazione mappa... {percentuale}%")
        progress.progress(percentuale)

        prompt = (
            "Rispondi SOLO con un JSON valido contenente i campi 'nodes' e 'edges'."
            " Includi nodes ed edges con campi 'from','to','relation'."
            f" Nodo centrale: '{central_node}'\n"
            f"\nBlocco {idx}/{totale_blocchi}:\n{b}"
        )
        resp = call_with_retries({"model": MODEL, "messages": [{"role": "user", "content": prompt}]})
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            lines = txt.splitlines()
            txt = "\n".join(lines[1:-1])
        start, end = txt.find('{'), txt.rfind('}') + 1
        raw = txt[start:end] if start != -1 and end != -1 else ''
        try:
            ris.append(json.loads(raw))
        except:
            st.warning(f"Parsing fallito per blocco {idx}")

    status_text.success("Mappa concettuale generata")
    progress.empty()

    raw_nodes = set()
    raw_edges = []
    for m in ris:
        for n in m.get('nodes', []):
            nid = n if isinstance(n, str) else n.get('id', '')
            if isinstance(nid, str):
                nid_str = nid.strip()
                if nid_str and not re.match(r'^(?:\d+|n\d+)$', nid_str, flags=re.IGNORECASE):
                    raw_nodes.add(nid_str)
        for e in m.get('edges', []):
            frm, to = e.get('from'), e.get('to')
            if frm in raw_nodes and to in raw_nodes:
                raw_edges.append({'from': frm, 'to': to, 'relation': e.get('relation', '')})

    tf = {n: len(re.findall(rf"\b{re.escape(n)}\b", testo, flags=re.IGNORECASE))
          for n in raw_nodes}

    index_terms = index_terms or []
    filtered_index = filtra_paragrafi_sottoparagrafi(index_terms)
    BOOST = 5
    for node in list(raw_nodes):
        for term in filtered_index:
            if re.search(rf"\b{re.escape(term)}\b", node, flags=re.IGNORECASE):
                tf[node] = tf.get(node, 0) + BOOST
                break

    return {'nodes': list(raw_nodes), 'edges': raw_edges, 'tf': tf, 'index_terms': filtered_index}

# === NUOVA FUNZIONE PER MATPLOTLIB ===

def draw_mind_map_from_json(mappa: dict, central_node: str, soglia: int):
    # 1) Filtra i nodi
    tf = mappa.get('tf', {})
    index_terms = set(mappa.get('index_terms', []))
    valid_nodes = {n for n, cnt in tf.items() if cnt >= soglia} | index_terms | {central_node}

    # 2) Seleziona solo edges di primo livello (da centrale)
    primo_livello = [e['to'] for e in mappa['edges']
                     if e['from'] == central_node and e['to'] in valid_nodes]

    # 3) Dividi i rami in due metà
    metà = len(primo_livello) // 2
    left_nodes  = primo_livello[:metà]
    right_nodes = primo_livello[metà:]

    # 4) Costruisci i sottorami
    def collect_subnodes(branch_nodes):
        d = {}
        for b in branch_nodes:
            subs = [e['to'] for e in mappa['edges']
                    if e['from'] == b and e['to'] in valid_nodes]
            d[b] = subs
        return d

    left_branches  = collect_subnodes(left_nodes)
    right_branches = collect_subnodes(right_nodes)

    # 5) Disegna con Matplotlib
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis("off")

    ax.text(0.5, 0.5, central_node,
            fontsize=16, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="lightblue"))

    def draw_branches(branches, direction="left"):
        if not branches:
            return
        step_y = 1 / (len(branches) + 1)
        for i, (branch, subnodes) in enumerate(branches.items()):
            y = 0.9 - i * step_y
            x = 0.25 if direction == "left" else 0.75
            ax.text(x, y, branch,
                    fontsize=12, ha="center", va="center",
                    bbox=dict(boxstyle="round", fc="lavender"))
            ax.plot([0.5, x], [0.5, y], "gray")

            # Sottorami
            substep = 0.04
            for j, sub in enumerate(subnodes):
                sub_y = y - (j+1) * substep
                ax.text(x, sub_y, f"- {sub}",
                        fontsize=10, ha="center", va="center",
                        bbox=dict(boxstyle="round", fc="white"))
                ax.plot([x, x], [y, sub_y], "gray")

    draw_branches(left_branches,  "left")
    draw_branches(right_branches, "right")

    plt.tight_layout()
    st.pyplot(fig)

# === STREAMLIT UI ===

st.set_page_config(page_title="Generatore Mappa Concettuale PDF", layout="wide")

col1, col2 = st.columns([5, 4])
with col1:
    st.title("Generatore Mappa Concettuale PDF")
with col2:
    st.empty()

doc = st.file_uploader("Carica il PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")
json_name = st.text_input("Nome JSON (senza estensione)", value="mappa_completa")
html_name = st.text_input("Nome file HTML (senza estensione)", value="grafico")
gif_path = "img/Progetto video 1.gif"
gif_placeholder = st.empty()

if st.button("Genera JSON completo") and doc:
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            gif_b64 = base64.b64encode(f.read()).decode("utf-8")
        img_html = f"""
        <div style="display:flex; justify-content:center; align-items:center; background:transparent;">
          <img src="data:image/gif;base64,{gif_b64}" style="max-width:300px; width:100%; height:auto;" alt="Loading..."/>
        </div>
        """
        gif_placeholder.markdown(img_html, unsafe_allow_html=True)
    else:
        gif_placeholder.markdown("<p style='text-align:center; color:red;'>GIF non trovata</p>", unsafe_allow_html=True)

    start_time = time.time()
    testo = estrai_testo_da_pdf(doc)
    index_terms = estrai_indice(testo)
    mappa = genera_mappa_concettuale(testo, central_node, index_terms=index_terms)
    elapsed = (time.time() - start_time) / 60
    gif_placeholder.empty()

    st.session_state['mappa'] = mappa
    st.session_state['testo'] = testo
    st.session_state['central_node'] = central_node
    st.session_state['index_terms'] = index_terms

    st.info(f"JSON generato in {elapsed:.2f} minuti")
    st.subheader("JSON Completo (con tf e termini indice)")
    st.json(mappa)
    json_bytes = json.dumps(mappa, ensure_ascii=False, indent=2).encode('utf-8')
    st.download_button("Scarica JSON", data=json_bytes, file_name=f"{json_name}.json", mime='application/json')

if 'mappa' in st.session_state:
    mappa = st.session_state['mappa']
    central_node = st.session_state['central_node']

    st.subheader("Visualizza mappa Matplotlib")
    soglia = st.number_input("Soglia occorrenze (tf ≥)", min_value=0, value=1, step=1)
    if st.button("Mostra mappa Matplotlib"):
        draw_mind_map_from_json(mappa, central_node, soglia)
