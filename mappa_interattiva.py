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
            "Rispondi SOLO con un JSON valido contenente i campi 'nodes' e 'edges'. "
            "Includi nodes ed edges con campi 'from','to','relation'. "
            "Nodo centrale: '" + central_node + "'\n"
            f"\nBlocco {idx}/{len(blocchi)}:\n{b}"
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
        progress.progress(idx / len(blocchi))
    progress.empty()
    st.success("Mappa concettuale generata")

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
            frm, to = e['from'], e['to']
            if frm in raw_nodes and to in raw_nodes:
                raw_edges.append({
                    'from': frm,
                    'to': to,
                    'relation': e.get('relation', '')
                })

    tf = {
        n: len(re.findall(rf"\b{re.escape(n)}\b", testo, flags=re.IGNORECASE))
        for n in raw_nodes
    }
    return {'nodes': list(raw_nodes), 'edges': raw_edges, 'tf': tf}

def crea_grafo_interattivo(mappa: dict, central_node: str, soglia: int) -> str:
    st.info(f"Creazione grafo con soglia >= {soglia}...")
    tf = mappa.get('tf', {})

    # 1) Nodi validi: sopra soglia + centrale
    valid_nodes = {n for n, cnt in tf.items() if cnt >= soglia} | {central_node}

    # 2) Grafo filtrato
    G_full = nx.DiGraph()
    G_full.add_nodes_from(valid_nodes)
    for e in mappa['edges']:
        frm, to = e['from'], e['to']
        if frm in valid_nodes and to in valid_nodes:
            G_full.add_edge(frm, to, relation=e.get('relation', ''))

    # 3) Discendenti del centrale
    reachable = {central_node}
    if central_node in G_full:
        reachable |= nx.descendants(G_full, central_node)

    # 4) Sotto‐grafo finale
    G = G_full.subgraph(reachable).copy()

    # 5) Community detection
    communities = list(nx.algorithms.community.louvain_communities(G.to_undirected()))
    group = {n: i for i, comm in enumerate(communities) for n in comm}

    # 6) Visualizzazione PyVis
    net = Network(directed=True, height='650px', width='100%')
    net.force_atlas_2based(
        gravity=-200,
        central_gravity=0.01,
        spring_length=800,
        spring_strength=0.001,
        damping=0.7
    )
    for n in G.nodes():
        size = 10 + (tf.get(n, 0) ** 0.5) * 20
        net.add_node(
            n, label=n, group=group.get(n, 0), size=size,
            x=0 if n == central_node else None,
            y=0 if n == central_node else None,
            fixed={'x': n == central_node, 'y': n == central_node}
        )
    for src, dst, data in G.edges(data=True):
        net.add_edge(src, dst, label=data.get('relation', ''))

    net.show_buttons(filter_=['physics', 'nodes', 'edges'])
    html_file = f"temp_graph_{int(time.time())}.html"
    net.save_graph(html_file)
    st.success("Grafo generato")
    return html_file

# === STREAMLIT APP ===
def main():
    st.title("Generatore di Mappa Concettuale")
    uploaded_file = st.file_uploader("Carica un PDF", type=["pdf"])
    # Nodo centrale fisso di default
    central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")

    if uploaded_file and central_node:
        # 1) Estrai testo e genera mappa + tf
        testo = estrai_testo_da_pdf(uploaded_file)
        mappa = genera_mappa_concettuale(testo, central_node)

        # 2) Mostro riepilogo tf per suggerire la soglia
        tf = mappa['tf']
        valori = sorted(set(tf.values()))
        st.write("Valori TF unici trovati nei nodi:", valori)
        st.write("Scegli una soglia fra questi valori (ad es. il valore mediano oppure un numero più alto per filtrare di più).")

        # 3) Input testuale per soglia
        soglia_str = st.text_input("Inserisci soglia di frequenza (tf)", value="1")
        try:
            soglia = int(soglia_str)
        except ValueError:
            st.error("La soglia deve essere un numero intero.")
            return

        # 4) Bottone per generare il grafo con la soglia scelta
        if st.button("Genera grafo"):
            html_path = crea_grafo_interattivo(mappa, central_node, soglia)
            components.html(open(html_path, "r").read(), height=650, width=800)

if __name__ == "__main__":
    main()
