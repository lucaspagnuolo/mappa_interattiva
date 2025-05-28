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


def genera_mappa_concettuale(testo: str, central_node: str) -> dict:
    blocchi = suddividi_testo(testo)
    ris = []
    st.info("Generazione mappa: elaborazione blocchi...")
    progress = st.progress(0)
    for idx, b in enumerate(blocchi, 1):
        prompt = (
            "Rispondi SOLO con un JSON valido contenente i campi 'nodes' e 'edges'."
            " Includi nodes ed edges con campi 'from','to','relation'."
            f" Nodo centrale: '{central_node}'\n\nBlocco {idx}/{len(blocchi)}:\n{b}"
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
        progress.progress(idx / len(blocchi))
    progress.empty()
    st.success("Mappa concettuale generata")

    # Aggrega nodi e archi validi
    raw_nodes = set()
    raw_edges = []
    for m in ris:
        for n in m.get('nodes', []):
            nid = n if isinstance(n, str) else n.get('id', '')
            nid_str = nid.strip()
            if nid_str and not re.match(r'^(?:\d+|n\d+)$', nid_str, flags=re.IGNORECASE):
                raw_nodes.add(nid_str)
        for e in m.get('edges', []):
            frm, to, rel = e.get('from'), e.get('to'), e.get('relation', '')
            if frm in raw_nodes and to in raw_nodes:
                raw_edges.append({'from': frm, 'to': to, 'relation': rel})

    # Calcola tf per nodi
    tf = {n: len(re.findall(rf"\b{re.escape(n)}\b", testo, flags=re.IGNORECASE)) for n in raw_nodes}

    # Calcola strength per relazioni
    # unica tripla (frm,to,rel)
    unique_rels = {(e['from'], e['to'], e['relation']) for e in raw_edges}
    rel_strength = {}
    for frm, to, rel in unique_rels:
        # cerca pattern "frm ... rel ... to"
        pattern = rf"\b{re.escape(frm)}\b.*?\b{re.escape(rel)}\b.*?\b{re.escape(to)}\b"
        count = len(re.findall(pattern, testo, flags=re.IGNORECASE | re.DOTALL))
        rel_strength[(frm, to, rel)] = count

    # Costruisci JSON finale
    return {
        'nodes': list(raw_nodes),
        'edges': raw_edges,
        'tf': tf,
        'relation_strength': [
            {'from': frm, 'to': to, 'relation': rel, 'count': rel_strength[(frm, to, rel)]}
            for frm, to, rel in unique_rels
        ]
    }


def crea_grafo_interattivo(mappa: dict, central_node: str, soglia: int) -> str:
    st.info(f"Creazione grafo con soglia >= {soglia}...")
    tf = mappa.get('tf', {})
    valid_nodes = {n for n, c in tf.items() if c >= soglia} | {central_node}
    G_full = nx.DiGraph()
    G_full.add_nodes_from(valid_nodes)
    for e in mappa['edges']:
        if e['from'] in valid_nodes and e['to'] in valid_nodes:
            G_full.add_edge(e['from'], e['to'], relation=e['relation'])
    reachable = {central_node}
    if central_node in G_full:
        reachable |= nx.descendants(G_full, central_node)
    G = G_full.subgraph(reachable).copy()

    # community detection
    communities = list(nx.algorithms.community.louvain_communities(G.to_undirected()))
    group = {n: i for i, comm in enumerate(communities) for n in comm}

    # relazione strength dict
        # relazione strength dict (usa get per sicurezza se chiave non esiste)
    rel_strength_dict = { (r['from'], r['to'], r['relation']): r.get('count', 0) \
                         for r in mappa.get('relation_strength', []) }
    net = Network(directed=True, height='650px', width='100%')
    net.force_atlas_2based(gravity=-200, central_gravity=0.01, spring_length=800, spring_strength=0.001, damping=0.7)
    for n in G.nodes():
        size = 10 + (tf.get(n,0)**0.5)*20
        net.add_node(n, label=n, group=group.get(n,0), size=size,
                     x=0 if n==central_node else None, y=0 if n==central_node else None,
                     fixed={'x':n==central_node,'y':n==central_node})
    for src,dst,data in G.edges(data=True):
        # calcola larghezza in base a strength
        key = (src,dst,data.get('relation',''))
        count = rel_strength_dict.get(key, 0)
        width = 1 + count*0.5
        net.add_edge(src, dst, label=f"{data.get('relation','')} ({count})", width=width)

    net.show_buttons(filter_=['physics','nodes','edges'])
    html_file = f"temp_graph_{int(time.time())}.html"
    net.save_graph(html_file)
    st.success("Grafo generato")
    return html_file

# === STREAMLIT UI ===
st.title("Generatore Mappa Concettuale PDF Interattivo")
# caricamento
doc = st.file_uploader("Carica il PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale", "Servizio di Manutenzione")
json_name = st.text_input("Nome JSON (senza estensione)", "mappa_completa")
html_name = st.text_input("Nome file HTML (senza estensione)", "grafico")

if st.button("Genera JSON completo") and doc:
    start = time.time()
    testo = estrai_testo_da_pdf(doc)
    mappa = genera_mappa_concettuale(testo, central_node)
    st.session_state['mappa'], st.session_state['testo'], st.session_state['central_node'] = mappa, testo, central_node
    st.info(f"JSON generato in {(time.time()-start)/60:.2f} minuti")
    st.json(mappa)
    st.download_button("Scarica JSON", json.dumps(mappa, ensure_ascii=False, indent=2), f"{json_name}.json")

if 'mappa' in st.session_state:
    st.subheader("Seleziona soglia per filtro nodo")
    soglia_str = st.text_input("Soglia occorrenze (numero intero)", value="1")
    soglia = None
    try:
        soglia = int(soglia_str)
    except ValueError:
        if soglia_str:
            st.error("Inserisci un numero intero valido per la soglia")
    if soglia is not None and st.button("Visualizza grafo con soglia"):
        html_file = crea_grafo_interattivo(
            st.session_state['mappa'],
            st.session_state['central_node'],
            soglia
        )
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        components.html(content, height=600)
        st.download_button("Scarica HTML", content, file_name=f"{html_name}_s{soglia}.html")
