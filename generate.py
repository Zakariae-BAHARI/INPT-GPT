import ollama

def generate_response(query, context):
    """Génère une réponse en utilisant Ollama avec le modèle Llama3.1."""
    prompt = f"""Réponds à la question suivante en utilisant les informations suivantes :

    Contexte :
    {context}

    Question :
    {query}
    """

    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
