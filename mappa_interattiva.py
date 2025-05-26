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

# Debug iniziale
st.write("Hello Mondo")

# === CONFIGURAZIONE API ===
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
MODEL = st.secrets.get("MISTRAL_MODEL", "mistral-large-latest")

# === FUNZIONI DI BACKEND ===
def estrai_testo_da_pdf(file) -> str:
    testo = []
    total = pdfplumber.open(file).pages.__len__()
    progress = st.progress(0)
    with pdfplumber.open(file) as pdf:
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
            " Obiettivo: Individuare i contesti e oggetti correlati e collegati al '" + central_node + "'."
            f"\nBlocco {idx}/{len(blocchi)}:\n{b}"
        )
        payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
        resp = call_with_retries(payload)
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
    # Estrai e pulisci nodes ed edges
    raw_nodes, raw_edges = set(), []
    for m in ris:
        for n in m.get('nodes', []):
            nid = n if isinstance(n, str) else n.get('id', '')
            raw_nodes.add(nid)
        for e in m.get('edges', []):
            raw_edges.append({
                'from': e.get('from'),
                'to': e.get('to'),
                'relation': e.get('relation', '')
            })
    # Filtra nodi numerici o placeholder tipo n1, n2...
    nodes = [n for n in raw_nodes if not re.match(r'^(?:\d+|n\d+)$', n)]
    edges = [e for e in raw_edges if e['from'] in nodes and e['to'] in nodes]
    return {'nodes': nodes, 'edges': edges}


def crea_grafo_interattivo(mappa: dict, testo: str, central_node: str, soglia: int) -> str:
    st.info("Creazione grafo interattivo...")
    progress = st.progress(0)
    tf = {n: len(re.findall(rf"\b{re.escape(n)}\b", testo, flags=re.IGNORECASE)) for n in mappa['nodes']}
    # Primi vicini e rimozione
    first_level = {e['to'] for e in mappa['edges'] if e['from'] == central_node}
    removed = {n for n in first_level if tf.get(n, 0) < soglia}
    queue = list(removed)
    while queue:
        cur = queue.pop()
        for e in mappa['edges']:
            if e['from'] == cur and e['to'] not in removed:
                removed.add(e['to'])
                queue.append(e['to'])
    filt_nodes = [n for n in mappa['nodes'] if n not in removed]
    filt_edges = [e for e in mappa['edges'] if e['from'] not in removed and e['to'] not in removed]
    G = nx.DiGraph()
    for i, n in enumerate(filt_nodes, 1):
        G.add_node(n)
        progress.progress(i / len(filt_nodes))
    for e in filt_edges:
        src, dst, rel = e['from'], e['to'], e.get('relation', '')
        G.add_edge(src, dst, relation=rel)
    progress.empty()
    communities = list(nx.algorithms.community.louvain_communities(G.to_undirected()))
    group = {n: i for i, comm in enumerate(communities) for n in comm}
    net = Network(directed=True, height='650px', width='100%')
    net.force_atlas_2based(
        gravity=-100,
        central_gravity=0.005,
        spring_length=800,
        spring_strength=0.002,
        damping=0.6
    )
    for n in G.nodes():
        size = 10 + (tf.get(n, 0) ** 0.5) * 20
        net.add_node(n, label=n, group=group.get(n, 0), size=size)
    for src, dst, data in G.edges(data=True):
        net.add_edge(src, dst, label=data.get('relation', ''))
    net.show_buttons(filter_=['physics', 'nodes', 'edges'])
    html_file = f"temp_graph_{int(time.time())}.html"
    net.save_graph(html_file)
    st.success("Grafo generato")
    return html_file

# === STREAMLIT UI ===
st.title("Generatore Mappa Concettuale PDF")

doc = st.file_uploader("Carica il file PDF", type=['pdf'])
central_node = st.text_input("Cosa vorresti analizzare?", value="Servizio di Manutenzione")
json_name = st.text_input("Nome file JSON (senza estensione)", value="mappa")
html_name = st.text_input("Nome file HTML (senza estensione)", value="grafico")

if doc and st.button("Genera mappa"):  # Prima fase
    testo = estrai_testo_da_pdf(doc)
    mappa = genera_mappa_concettuale(testo, central_node)
    # Calcola TF e ordina
    tf = {n: len(re.findall(rf"\b{re.escape(n)}\b", testo, flags=re.IGNORECASE)) for n in mappa['nodes']}
    sorted_tf = sorted(tf.items(), key=lambda x: x[1], reverse=True)
    st.subheader("Frequenza termini (TF)")
    for nodo, freq in sorted_tf:
        st.write(f"{nodo}: {freq}")
        # Campo soglia basata su TF massimo
    max_tf = sorted_tf[0][1] if sorted_tf else 1
    soglia = st.number_input("Imposta la soglia sulla base delle frequenze visibili", min_value=1, max_value=max_tf, value=1, step=1)
    # Bottone per generare il grafo
    if st.button("Conferma soglia e procedi a grafo"):
        st.session_state['testo'] = testo
        st.session_state['mappa'] = mappa
        st.session_state['soglia'] = soglia

if 'mappa' in st.session_state and st.button("Genera grafo interattivo")['testo'] = testo
    st.session_state['mappa'] = mappa
    st.session_state['soglia'] = soglia

if 'mappa' in st.session_state and st.button("Genera grafo interattivo"):
    html_file = crea_grafo_interattivo(
        st.session_state['mappa'], st.session_state['testo'], central_node, st.session_state['soglia']
    )
    # Download JSON
    json_bytes = json.dumps(st.session_state['mappa'], ensure_ascii=False, indent=2).encode('utf-8')
    st.download_button("Scarica JSON", data=json_bytes, file_name=f"{json_name}.json", mime='application/json')
    # Anteprima e download HTML
    st.subheader("Anteprima Grafico Interattivo")
    html_content = open(html_file, 'r', encoding='utf-8').read()
    components.html(html_content, height=600, scrolling=True)
    st.download_button("Scarica Grafico HTML", data=html_content, file_name=f"{html_name}.html", mime='text/html')
