import os
import json
import re
import time
import pdfplumber
import networkx as nx
from math import cos, sin
from mistralai import Mistral, SDKError
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components  # ← questa riga mancava
import base64
from PIL import Image  # serve solo per essere sicuri di leggere il file senza errori

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

        # Layout radiale a più livelli: nodo centrale al centro, neighbor e discendenti su cerchi concentrici
    depth = nx.single_source_shortest_path_length(G, central_node)
    levels = {}
    for n, d in depth.items():
        levels.setdefault(d, []).append(n)
    max_level = max(levels.keys()) if levels else 1
    from random import uniform  # per piccolo jitter angolare
    ring_radius = 300  # distanza di base tra livelli (aumentato per evitare sovrapposizioni)
    positions = {}
    positions[central_node] = (0, 0)
    for lvl, nodes_at_lvl in levels.items():
        # cerchio di livello, con raggio maggiore per livelli più esterni
        num_n = len(nodes_at_lvl)
        radius_lvl = lvl * ring_radius
        for idx, node in enumerate(nodes_at_lvl):
            # angolo uniforme + piccolo jitter
            base_angle = 2 * 3.141592653589793 * idx / num_n
            angle = base_angle + uniform(-0.1, 0.1)
            x = radius_lvl * cos(angle)
            y = radius_lvl * sin(angle)
            positions[node] = (x, y)
        if lvl == 0:
            continue
        num_n = len(nodes_at_lvl)
        radius_lvl = lvl * ring_radius
        for idx, node in enumerate(nodes_at_lvl):
            angle = 2 * 3.141592653589793 * idx / num_n
            x = radius_lvl * cos(angle)
            y = radius_lvl * sin(angle)
            positions[node] = (x, y)

    net = Network(directed=True, height='650px', width='100%')
    net.toggle_physics(False)  # disabilita la fisica per layout fisso
    for n in G.nodes():
        size = 10 + (tf.get(n, 0) ** 0.5) * 20
        x, y = positions.get(n, (None, None))
        net.add_node(
            n,
            label=n,
            x=x,
            y=y,
            fixed={'x': True, 'y': True},
            size=size
        )
    for src, dst, data in G.edges(data=True):
        net.add_edge(src, dst, label=data.get('relation', ''))
    net.show_buttons(filter_=['nodes', 'edges'])
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

# 1) Caricamento PDF e parametri base

doc = st.file_uploader("Carica il PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")
json_name = st.text_input("Nome JSON (senza estensione)", value="mappa_completa")
html_name = st.text_input("Nome file HTML (senza estensione)", value="grafico")

# 2) Path della GIF (non serve ricavarne dimensioni con PIL)
gif_path = "img/Progetto video 1.gif"
if not os.path.exists(gif_path):
    st.warning("GIF non trovata: controlla che il file esista in img/Progetto video 1.gif")

# 3) Placeholder per la GIF
gif_placeholder = st.empty()

# 4) Bottone "Genera JSON completo"
if st.button("Genera JSON completo") and doc:
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            gif_bytes = f.read()
        gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")
        img_html = f"""
        <div style="display:flex; justify-content:center; align-items:center; background:transparent; margin:0; padding:0;">
          <img 
            src="data:image/gif;base64,{gif_b64}"
            style="
              max-width:300px;
              width:100%;
              height:auto;
              display:block;
              margin:0;
              padding:0;
            "
            alt="Loading..."
          />
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

# 5) Generazione grafo interattivo
if 'mappa' in st.session_state:
    mappa = st.session_state['mappa']
    central_node = st.session_state.get('central_node', central_node)
    st.subheader("Seleziona soglia per filtro nodo")
    soglia_input = st.text_input("Soglia occorrenze (numero intero)", value="1")
    if st.button("Visualizza grafo con soglia"):
        try:
            soglia = int(soglia_input)
            start_time = time.time()
            html_file = crea_grafo_interattivo(mappa, central_node, soglia)
            elapsed = (time.time() - start_time) / 60
            st.info(f"Grafo generato in {elapsed:.2f} minuti (soglia >= {soglia})")
            st.subheader(f"Grafo (soglia >= {soglia})")
            content = open(html_file, 'r', encoding='utf-8').read()
            components.html(content, height=600, scrolling=True)
            st.download_button("Scarica HTML", data=content, file_name=f"{html_name}_s{soglia}.html", mime='text/html')
        except ValueError:
            st.error("Inserisci un numero intero valido per la soglia.")
