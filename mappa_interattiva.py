# === STREAMLIT UI ===
st.title("Generatore Mappa Concettuale PDF Interattivo")

doc = st.file_uploader("Carica il PDF", type=['pdf'])
central_node = st.text_input("Nodo centrale", value="Servizio di Manutenzione")
json_name = st.text_input("Nome JSON (senza estensione)", value="mappa_completa")
html_name = st.text_input("Nome file HTML (senza estensione)", value="grafico")

# 1) Generazione JSON completo
if st.button("Genera JSON completo") and doc:
    start_time = time.time()
    testo = estrai_testo_da_pdf(doc)
    mappa = genera_mappa_concettuale(testo, central_node)
    st.session_state['mappa'] = mappa
    st.session_state['testo'] = testo
    st.session_state['central_node'] = central_node
    elapsed = (time.time() - start_time) / 60
    st.info(f"JSON generato in {elapsed:.2f} minuti")
    st.subheader("JSON Completo (con tf)")
    st.json(mappa)
    json_bytes = json.dumps(mappa, ensure_ascii=False, indent=2).encode('utf-8')
    st.download_button("Scarica JSON", data=json_bytes, file_name=f"{json_name}.json", mime='application/json')

# 2) Mostra input soglia e creazione grafo SOLO se JSON generato
if 'mappa' in st.session_state:
    mappa = st.session_state['mappa']
    central_node = st.session_state['central_node']
    # Input soglia appare solo dopo JSON
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
