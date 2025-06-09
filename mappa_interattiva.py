# -*- coding: utf-8 -*-
import os
import json
import re
import time
import pdfplumber
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from mistralai import Mistral, SDKError
from streamlit_agraph import agraph, Config, Node, Edge

# =============================================
# CONFIGURAZIONE API
# =============================================
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
MODEL = st.secrets.get("MISTRAL_MODEL", "mistral-large-latest")

# =============================================
# FUNZIONI DI BACKEND
# =============================================
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
            "Rispondi SOLO con un JSON valido contenente i campi 'nodes' e 'edges'. "
            "Includi nodes ed edges con campi 'from','to','relation'. "
            f"Nodo centrale: '{central_node}'\n\n"
            f"Blocco {idx}/{totale_blocchi}:\n{b}"
        )
        resp = call_with_retries({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}]
        })
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

# =============================================
# STREAMLIT UI
# =============================================
st.set_page_config(page_title="Generatore Mappa Concettuale PDF – Layout Radiale", layout="wide")
st.title("Generatore Mappa Concettuale PDF – Layout Radiale")

pdf_file = st.file_uploader("Carica il PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale (omesso in visualizzazione)", value="Servizio di Manutenzione")

if pdf_file and st.button("Genera Mappa e Visualizza"):
    testo = estrai_testo_da_pdf(pdf_file)
    idx_terms = estrai_indice(testo)
    mappa = genera_mappa_concettuale(testo, central_node, index_terms=idx_terms)
    st.session_state['mappa'] = mappa

if 'mappa' in st.session_state:
    mappa = st.session_state['mappa']
    tf = mappa['tf']
    edges_raw = mappa['edges']

    G = nx.DiGraph()
    for n in tf:
        if n != central_node:
            G.add_node(n)
    for e in edges_raw:
        if e['from'] != central_node and e['to'] != central_node:
            G.add_edge(e['from'], e['to'], relation=e.get('relation',''))

    nodes = [
        Node(id=n, label=n, size=10 + (tf.get(n,0)**0.5)*20)
        for n in G.nodes()
    ]
    edges = [
        Edge(source=src, target=dst, label=data.get('relation',''))
        for src, dst, data in G.edges(data=True)
    ]

    config = Config(
        width="100%", height=700, directed=True,
        layout='radial',          # <-- LAYOUT RADIALE SEMPRE
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        initialZoom=1.0,
        node={'fontSize': 12}
    )

    st.subheader("Mappa Concettuale – Radiale")
    agraph(nodes=nodes, edges=edges, options=config)

    st.download_button(
        "Scarica JSON",
        data=json.dumps(mappa, ensure_ascii=False, indent=2),
        file_name="mappa_completa.json",
        mime='application/json'
    )

    # Se vuoi esportare in HTML con pyvis, scommenta e installa pyvis:
    # from pyvis.network import Network
    #
    # def salva_mappa_html(nodes, edges, filename="mappa_radiale.html"):
    #     net = Network(height="750px", width="100%", directed=True)
    #     for node in nodes:
    #         net.add_node(node.id, label=node.label, size=node.size)
    #     for edge in edges:
    #         net.add_edge(edge.source, edge.target, label=edge.label)
    #     net.show(filename)
    #     return filename
    #
    # if st.button("Esporta come HTML"):
    #     html_file = salva_mappa_html(nodes, edges)
    #     with open(html_file, "r", encoding="utf-8") as f:
    #         html_content = f.read()
    #     st.download_button("Scarica HTML", data=html_content,
    #                        file_name="mappa_radiale.html", mime="text/html")
