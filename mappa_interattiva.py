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
json
{{\"{central_node}\": {{\"Ramo A\": [\"Sottoramo1\", ...], ...}}}}

Nessun altro testo.
Estrai dai seguenti contenuti:
{block_text}
"""
    resp = call_with_retries({'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}]})
    txt = resp.choices[0].message.content
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.S)
    if not m:
        m = re.search(r"```(?:json|python)?\s*(\{.*?\})\s*```", txt, flags=re.S)
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

# Funzione di disegno di base
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
        if n > 1:
            ys = [0.9 - i * (0.8 / (n - 1)) for i in range(n)]
        else:
            ys = [0.5]
        return {item[0]: y for item, y in zip(flat, ys)}

    posL, posR = compute_pos(flat_L), compute_pos(flat_R)
    fig, ax = plt.subplots(figsize=(20, max(12, (len(flat_L)+len(flat_R)+1)*0.3)))
    ax.axis('off')
    total_nodes = len(flat_L) + len(flat_R) + 1
    main_fs = 16 if total_nodes <= 50 else max(8, 16 * 50 / total_nodes)
    sub_fs = main_fs * 0.8

    # Nodo centrale
    ax.text(0.5, 0.5, central_node, fontsize=main_fs, ha='center', va='center', bbox=dict(boxstyle='round', fc='lightblue'))

    # Connessioni
    for node, depth, *rest in flat_L:
        if depth == 0:
            ax.plot([0.5, 0.25], [0.5, posL[node]], 'gray')
    for node, depth, *rest in flat_R:
        if depth == 0:
            ax.plot([0.5, 0.75], [0.5, posR[node]], 'gray')

    # Etichette
    for node, depth, *rest in flat_L:
        x = 0.25 - 0.03 * depth
        y = posL[node]
        fs = main_fs if depth == 0 else sub_fs
        txt = node if depth == 0 else f"- {node}"
        ax.text(x, y, txt, fontsize=fs, ha='center', va='center', bbox=dict(boxstyle='round', fc='lavender' if depth == 0 else 'white'))
        if depth == 1:
            ax.plot([0.25, 0.25], [posL[rest[0]], y], 'gray')
    for node, depth, *rest in flat_R:
        x = 0.75 + 0.03 * depth
        y = posR[node]
        fs = main_fs if depth == 0 else sub_fs
        txt = node if depth == 0 else f"- {node}"
        ax.text(x, y, txt, fontsize=fs, ha='center', va='center', bbox=dict(boxstyle='round', fc='lavender' if depth == 0 else 'white'))
        if depth == 1:
            ax.plot([0.75, 0.75], [posR[rest[0]], y], 'gray')

    plt.tight_layout()
    st.pyplot(fig)

# Funzioni batch

def draw_mind_map_subset(central_node: str, branches: dict[str, set[str]], subset_primari: list[str]):
    sub_branches = {r: branches[r] for r in subset_primari if r in branches}
    draw_mind_map(central_node, {**{central_node: set()}, **sub_branches})


def draw_mind_maps_in_batches(central_node: str, branches: dict[str, set[str]], batch_size: int = 10):
    primari = list(branches.keys())
    batches = [primari[i:i+batch_size] for i in range(0, len(primari), batch_size)]
    for i, batch in enumerate(batches, 1):
        st.markdown(f"### Mappa batch {i}/{len(batches)} (rami: {len(batch)})")
        draw_mind_map_subset(central_node, branches, batch)

# === MAPPA CIRCOLARE ===
def estrai_indice(testo: str) -> list[str]:
    righe = testo.splitlines()
    try:
        start = next(i for i, r in enumerate(righe) if re.match(r'^(Indice|Sommario)\b', r, re.IGNORECASE))
    except StopIteration:
        return []
    termini = []
    for r in righe[start+1:]:
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
    return [t for t in index_terms if re.match(r'^\d+(?:\.\d+)*\s+[A-ZÀ-ÖØ-Ý]', t)]


def genera_mappa_concettuale(testo: str, central_node: str, index_terms: list[str] = None) -> dict:
    blocchi = suddividi_testo(testo)
    ris = []
    status = st.empty()
    progress = st.progress(0)
    totale = len(blocchi)
    for idx, b in enumerate(blocchi, 1):
        pct = int(((idx - 1) / totale) * 100)
        status.info(f"Generazione mappa... {pct}%")
        progress.progress(pct)
        prompt = (
            "Rispondi SOLO con un JSON valido contenente i campi 'nodes' e 'edges'."
            " Includi nodes ed edges con campi 'from','to','relation'."
            f" Nodo centrale: '{central_node}'\n\nBlocco {idx}/{totale}:\n{b}"
        )
        resp = call_with_retries({'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}]})
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            lines = txt.splitlines()
            txt = "\n".join(lines[1:-1])
        start, end = txt.find('{'), txt.rfind('}') + 1
        raw = txt[start:end] if start != -1 and end != -1 else ''
        try:
            ris.append(json.loads(raw))
        except json.JSONDecodeError:
            st.warning(f"Parsing fallito per blocco {idx}")
    status.success("Mappa concettuale generata")
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
            frm = e.get('from')
            to = e.get('to')
            if frm in raw_nodes and to in raw_nodes:
                raw_edges.append({'from': frm, 'to': to, 'relation': e.get('relation', '')})

    tf = {
        n: len(re.findall(rf"\b{re.escape(n)}\b", testo, flags=re.IGNORECASE))
        for n in raw_nodes
    }
    idxs = index_terms or []
    filt = filtra_paragrafi_sottoparagrafi(idxs)
    BOOST = 5
    for node in list(raw_nodes):
        if any(re.search(rf"\b{re.escape(term)}\b", node, flags=re.IGNORECASE) for term in filt):
            tf[node] = tf.get(node, 0) + BOOST

    return {'nodes': list(raw_nodes), 'edges': raw_edges, 'tf': tf, 'index_terms': filt}


def crea_grafo_interattivo(mappa: dict, central_node: str, soglia: int) -> str:
    st.info(f"Creazione grafo con soglia >= {soglia}...")
    tf = mappa.get('tf', {})
    idxs = set(mappa.get('index_terms', []))
    valid = {n for n, c in tf.items() if c >= soglia} | idxs | {central_node}
    G_full = nx.DiGraph()
    G_full.add_nodes_from(valid)
    for e in mappa['edges']:
        frm = e['from']
        to = e['to']
        if frm in valid and to in valid:
            G_full.add_edge(frm, to, relation=e.get('relation', ''))
    reachable = {central_node}
    if central_node in G_full:
        reachable |= nx.descendants(G_full, central_node)
    G = G_full.subgraph(reachable).copy()
    comms = list(nx.algorithms.community.louvain_communities(G.to_undirected()))
    group = {n: i for i, comm in enumerate(comms) for n in comm}

    net = Network(directed=True, height='650px', width='100%')
    net.force_atlas_2based(gravity=-200, central_gravity=0.01, spring_length=800, spring_strength=0.001, damping=0.7)
    for n in G.nodes():
        size = 10 + (tf.get(n, 0) ** 0.5) * 20
        net.add_node(n, label=n, group=group.get(n, 0), size=size,
                     x=0 if n == central_node else None,
                     y=0 if n == central_node else None,
                     fixed={'x': n == central_node, 'y': n == central_node})
    for src, dst, data in G.edges(data=True):
        net.add_edge(src, dst, label=data.get('relation', ''))

    net.show_buttons(filter_=['physics', 'nodes', 'edges'])
    html_file = f"temp_graph_{int(time.time())}.html"
    net.save_graph(html_file)
    st.success("Grafo generato")
    return html_file

# === STREAMLIT APP ===
st.set_page_config(page_title="Generatore Mappa Concettuale PDF", layout="wide")
mode = st.sidebar.selectbox("Seleziona modalità", ["Mappa Gerarchica", "Mappa Circolare"])

doc = st.file_uploader("Carica il PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")

if mode == "Mappa Gerarchica":
    if st.button("Genera Mappa Gerarchica") and doc:
        testo = estrai_testo_da_pdf(doc)
        blocchi = suddividi_testo(testo)
        accum = {}
        status = st.empty(); prog = st.progress(0)
        total = len(blocchi)
        for idx, blk in enumerate(blocchi, 1):
            pct = int(idx/total*100); status.info(f"Generazione... {pct}%"); prog.progress(pct)
            part = genera_struttura_per_blocco(blk, central_node)
            if not part: part = {"[automatic]": [central_node]}
            merge_structures(accum, part)
        prog.empty(); status.success("Mappa Gerarchica completata!")
        st.session_state['accum'] = accum

    if 'accum' in st.session_state:
        batch_size = st.sidebar.slider("Rami per immagine", 1, 20, 10)
        draw_mind_maps_in_batches(central_node, st.session_state['accum'], batch_size)
else:
    if st.button("Genera JSON Completo") and doc:
        gif_ph = st.empty()
        gif_path = "img/Progetto video 1.gif"
        if os.path.exists(gif_path):
            b64 = base64.b64encode(open(gif_path, 'rb').read()).decode()
            gif_ph.markdown(f"<img src='data:image/gif;base64,{b64}' width=200/>", unsafe_allow_html=True)
        testo = estrai_testo_da_pdf(doc)
        index_terms = estrai_indice(testo)
        st.session_state['mappa'] = genera_mappa_concettuale(testo, central_node, index_terms)
        st.session_state['testo'] = testo
        st.session_state['central_node'] = central_node
        st.session_state['index_terms'] = index_terms
        gif_ph.empty()
        st.subheader("JSON Completo")
        st.json(st.session_state['mappa'])
    if 'mappa' in st.session_state:
        soglia = st.number_input("Soglia occorrenze (numero intero)", min_value=1, value=1)
        if st.button("Visualizza Grafo"):
            html_file = crea_grafo_interattivo(st.session_state['mappa'], st.session_state['central_node'], soglia)
            content = open(html_file, 'r', encoding='utf-8').read()
            components.html(content, height=600, scrolling=True)
            st.download_button("Scarica HTML", data=content, file_name=f"grafico_s{soglia}.html", mime='text/html')
