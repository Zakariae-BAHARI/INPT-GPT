import streamlit as st
from retrieval import SKLearnVectorStore
from generate import generate_response

# Chargement du logo d'Orange
logo_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Orange_logo.svg"

# Initialisation de la base de connaissances
retriever = SKLearnVectorStore()
retriever.load_data()

# Définition du style CSS
st.markdown(
    """
    <style>
        body {
            background-color: #000;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #ff7900;
            text-align: center;
        }
        .question-box {
            background-color: #fff;
            padding: 10px;
            border-radius: 10px;
        }
        .response-box {
            background-color: #ff7900;
            color: white;
            padding: 15px;
            border-radius: 10px;
        }
        .source-box {
            background-color: #fff;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Affichage du logo et du titre
st.image(logo_url, width=150)
st.markdown('<h1 class="title">🔎 Orange info Hub</h1>', unsafe_allow_html=True)

# Champ de saisie de la question
query = st.text_input("Posez votre question :")

if query:
    results = retriever.search(query)
    context = "\n\n".join([doc["content"] for doc in results])

    response = generate_response(query, context)

    # Affichage de la réponse générée
    st.markdown('<h2 style="color:#ff7900;">📢 Réponse Générée</h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)

    # Affichage des sources utilisées
    st.markdown('<h2 style="color:#ff7900;">📚 Sources Utilisées</h2>', unsafe_allow_html=True)
    for doc in results:
        st.markdown(f'<div class="source-box">🔗 <a href="{doc["url"]}" target="_blank">{doc["title"]}</a></div>', unsafe_allow_html=True)
