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
            " Nodo centrale: '" + central_node + "'\n"
            f"\nBlocco {idx}/{len(blocchi)}:\n{b}"
        )
        resp = call_with_retries({"model": MODEL, "messages": [{"role": "user", "content": prompt}]})
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            lines = txt.splitlines(); txt = "\n".join(lines[1:-1])
        start, end = txt.find('{'), txt.rfind('}') + 1
        raw = txt[start:end] if start != -1 and end != -1 else ''
        try:
            ris.append(json.loads(raw))
        except:
            st.warning(f"Parsing fallito per blocco {idx}")
        progress.progress(idx / len(blocchi))
    progress.empty(); st.success("Mappa concettuale generata")

    raw_nodes, raw_edges = set(), []
    for m in ris:
        for n in m.get('nodes', []):
            nid = n if isinstance(n, str) else n.get('id', '')
            if isinstance(nid, str) and nid.strip(): raw_nodes.add(nid.strip())
        for e in m.get('edges', []): raw_edges.append({'from': e['from'], 'to': e['to'], 'relation': e.get('relation','')})

    tf = {n: len(re.findall(rf"\b{re.escape(n)}\b", testo, flags=re.IGNORECASE)) for n in raw_nodes}
    return {'nodes': list(raw_nodes), 'edges': raw_edges, 'tf': tf}


def crea_grafo_interattivo(mappa: dict, central_node: str, soglia: int) -> str:
    st.info(f"Creazione grafo con soglia >= {soglia}...")
    tf = mappa['tf']; first = {e['to'] for e in mappa['edges'] if e['from']==central_node}
    rem = {n for n in first if tf.get(n,0)<soglia}; queue=list(rem)
    while queue:
        cur=queue.pop()
        for e in mappa['edges']:
            if e['from']==cur and e['to'] not in rem: rem.add(e['to']); queue.append(e['to'])
    nodes = [n for n in mappa['nodes'] if n not in rem]
    edges = [e for e in mappa['edges'] if e['from'] not in rem and e['to'] not in rem]
    G=nx.DiGraph(); G.add_nodes_from(nodes); G.add_edges_from([(e['from'],e['to'],{'relation':e['relation']}) for e in edges])
    comm = list(nx.algorithms.community.louvain_communities(G.to_undirected()))
    group={n:i for i,com in enumerate(comm) for n in com}
    net=Network(directed=True,height='650px',width='100%')
    net.force_atlas_2based(gravity=-200,central_gravity=0.01,spring_length=800,spring_strength=0.001,damping=0.7)
    for n in G.nodes(): size=10+(tf.get(n,0)**0.5)*20; net.add_node(n,label=n,group=group[n],size=size, x=0 if n==central_node else None, y=0 if n==central_node else None, fixed={'x':n==central_node,'y':n==central_node})
    for u,v,data in G.edges(data=True): net.add_edge(u,v,label=data['relation'])
    net.show_buttons(filter_=['physics','nodes','edges'])
    file=f"temp_graph_{int(time.time())}.html"; net.save_graph(file); st.success("Grafo generato")
    return file

# === STREAMLIT UI ===
st.title("Generatore Mappa Concettuale PDF Interattivo")
doc = st.file_uploader("Carica il file PDF",type=['pdf'])
central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")
json_name = st.text_input("Nome JSON", value="mappa_completa")
html_name = st.text_input("Nome HTML", value="grafico")

# Genera JSON completo
if st.button("Genera JSON completo") and doc:
    testo = estrai_testo_da_pdf(doc)
    mappa = genera_mappa_concettuale(testo, central_node)
    st.session_state['mappa']=mappa; st.session_state['central_node']=central_node
    st.subheader("JSON Completo (con tf)"); st.json(mappa)
    b=json.dumps(mappa,ensure_ascii=False,indent=2).encode('utf-8')
    st.download_button("Scarica JSON",data=b,file_name=f"{json_name}.json",mime='application/json')

# Mostra input soglia e grafo SOLO dopo JSON
after = 'mappa' in st.session_state
if after:
    mappa=st.session_state['mappa']; central_node=st.session_state['central_node']
    soglia_input=st.text_input("Soglia occorrenze (numero intero)",value="1")
    if soglia_input:
        try:
            soglia=int(soglia_input)
            if st.button("Visualizza grafo con soglia"):
                file=crea_grafo_interattivo(mappa,central_node,soglia)
                st.subheader(f"Grafo con soglia >= {soglia}")
                html=open(file,'r',encoding='utf-8').read(); components.html(html,height=600,scrolling=True)
                st.download_button("Scarica HTML",data=html,file_name=f"{html_name}_s{str(soglia)}.html",mime='text/html')
        except:
            st.error("Inserisci un numero intero valido per la soglia.")
