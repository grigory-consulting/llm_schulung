import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import requests   # Für Ollama-API

# 1. Patientenprofil-Kontexte reinladen
df = pd.read_csv("patient_rag_chunks.csv")
docs = df['profile_context'].tolist()
ids  = df['patient_id'].tolist()
names = df['name'].tolist()

# 2. Embedding vorbereiten
sb_embed = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="patients_demo", embedding_function=sb_embed)

# 3. Indexaufbau (nur einmalig nötig)
if not collection.count():
    collection.add(documents=docs, ids=ids, metadatas=[{"name": n} for n in names])

# 4. Beispielabfrage
query_text = "Patient with high income and low healthcare expenses"
# Suche n beste Kandidaten
results = collection.query(
    query_texts=[query_text],
    n_results=1,
    include=["documents", "metadatas"]
)



# Top-Dokument extrahieren (ava for RAG)
top_doc = results['documents'][0][0]

print(top_doc)


# 5. Ollama-Gemma Prompting für finale Antwort:
def ask_gemma(system_prompt, user_prompt, model="gemma3:27b", url="http://localhost:11434/api/chat"):
    d = {
        "model": model, 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
    }
    resp = requests.post(url, json=d)
    return resp.json()["message"]["content"]

rag_prompt = f"""
Here is the current patient profile. Please pay special attention to important medical indicators.

{top_doc}

Question: Please interpret the medical findings, particularly cholesterol, BMI, blood pressure, and stress factors. What would you recommend to the patient?
"""

system = "You are a medical expert and provide clear, well-structured, and accessible explanations and suggestions."
gemma_output = ask_gemma(system, rag_prompt, model="gemma3:27b")
print("Gemma's answer:\n", gemma_output)

