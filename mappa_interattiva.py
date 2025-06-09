import os
import json
import re
import time
import pdfplumber
import networkx as nx
from mistralai import Mistral, SDKError
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import base64

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
        start = next(i for i, r in enumerate(righe) if re.match(r'^(Indice|Sommario)\b', r, re.IGNORECASE))
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

    tf = {n: len(re.findall(rf"\b{re.escape(n)}\b", testo, flags=re.IGNORECASE)) for n in raw_nodes}

    index_terms = index_terms or []
    filtered_index = filtra_paragrafi_sottoparagrafi(index_terms)
    BOOST = 5
    for node in list(raw_nodes):
        for term in filtered_index:
            if re.search(rf"\b{re.escape(term)}\b", node, flags=re.IGNORECASE):
                tf[node] = tf.get(node, 0) + BOOST
                break

    return {'nodes': list(raw_nodes), 'edges': raw_edges, 'tf': tf, 'index_terms': filtered_index}


def crea_grafo_interattivo(mappa: dict, central_node: str, soglia: int) -> str:
    st.info(f"Creazione grafo con soglia >= {soglia}...")
    tf = mappa.get('tf', {})
    index_terms = set(mappa.get('index_terms', []))
    valid_nodes = {n for n, count in tf.items() if count >= soglia} | index_terms | {central_node}

    G_full = nx.DiGraph()
    G_full.add_nodes_from(valid_nodes)
    for e in mappa['edges']:
        frm, to = e.get('from'), e.get('to')
        if frm in valid_nodes and to in valid_nodes:
            G_full.add_edge(frm, to, relation=e.get('relation', ''))

    reachable = {central_node}
    if central_node in G_full:
        reachable |= nx.descendants(G_full, central_node)
    G = G_full.subgraph(reachable).copy()

    try:
        communities = list(nx.algorithms.community.louvain_communities(G.to_undirected()))
        group_map = {n: i for i, comm in enumerate(communities) for n in comm}
    except Exception:
        group_map = {n: 0 for n in G.nodes()}

    net = Network(directed=True, height='650px', width='100%')
    net.toggle_physics(False)
    options = {
        'layout': {'hierarchical': {'enabled': True, 'direction': 'LR', 'sortMethod': 'hubsize'}},
        'physics': {'hierarchicalRepulsion': {'nodeDistance': 200}}
    }
    net.set_options(json.dumps({'layout': options['layout'], 'physics': options['physics']}))

    for n in G.nodes():
        size = 10 + (tf.get(n, 0)**0.5) * 20
        net.add_node(n, label=n, size=size, group=group_map.get(n, 0))
    for src, dst, data in G.edges(data=True):
        net.add_edge(src, dst, label=data.get('relation', ''))

    html_file = f"temp_graph_{int(time.time())}.html"
    net.save_graph(html_file)
    st.success("Grafo generato")
    return html_file

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
if not os.path.exists(gif_path):
    st.warning("GIF non trovata: controlla che il file esista in img/Progetto video 1.gif")

gif_placeholder = st.empty()

if st.button("Genera JSON completo") and doc:
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            gif_bytes = f.read()
        gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")
        img_html = (
            f"<div style='display:flex;justify-content:center;align-items:center;'>"
            f"<img src='data:image/gif;base64,{gif_b64}' "\
