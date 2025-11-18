# -*- coding: utf-8 -*-
"""
rag_embeddings_demo.py
----------------------
Minimaler RAG-Flow mit EMBEDDINGS und *fixem* Threshold (kein TF-IDF, kein dynamisches Gating).

Was das Skript zeigt:
- Langer Demo-Text (Energie-Themen) -> Chunking mit Overlap
- Embeddings (SentenceTransformer lokal ODER OpenAI)
- Cosine-Similarity gegen alle Chunks
- *Fester* Relevanz-Threshold: Nur wenn Top-1 >= ACCEPT_THRESHOLD wird Kontext/Prompt/LLM gezeigt
- Optional: LLM-Antwort (OpenAI / LM Studio), sonst nur Retrieval-Demo

Benutzung:
    # nur lokale Embeddings, ohne LLM (Default)
    python rag_embeddings_demo.py

Optionale Umgebungsvariablen:
    export EMBED_BACKEND=local             # 'local' | 'openai'
    export GEN_BACKEND=none                # 'none'  | 'openai' | 'lmstudio'
    export OPENAI_API_KEY=sk-...           # falls OpenAI/LM Studio genutzt wird
    export OPENAI_BASE_URL=http://localhost:1234/v1  # für LM Studio
    export ACCEPT_THRESHOLD=0.30           # fester Cosine-Schwellenwert (0..1)
    export TOP_K=5
    export MAX_WORDS_PER_CHUNK=120
    export OVERLAP_SENTENCES=2
"""
import os
import re
from typing import List, Dict, Tuple
import numpy as np

# -----------------------------
# Konfiguration
# -----------------------------
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local")          # 'local' | 'openai'
GEN_BACKEND   = "lmstudio"           # 'none' | 'openai' | 'lmstudio'

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL      = "http://localhost:1234/v1"         # für LM Studio im OpenAI-Modus
OPENAI_EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL    = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
LMSTUDIO_CHAT_MODEL  = os.getenv("LMSTUDIO_CHAT_MODEL", "qwen/qwen3-30b-a3b_q4")

TOP_K                = int(os.getenv("TOP_K", "5"))
ACCEPT_THRESHOLD     = float(os.getenv("ACCEPT_THRESHOLD", "0.30"))  

MAX_WORDS_PER_CHUNK  = int(os.getenv("MAX_WORDS_PER_CHUNK", "120"))
OVERLAP_SENTENCES    = int(os.getenv("OVERLAP_SENTENCES", "2"))

# -----------------------------
# Optionaler OpenAI-Client (auch für LM Studio/OpenAI-kompatibel)
# -----------------------------
def _openai_client():
    from openai import OpenAI
    if OPENAI_BASE_URL:
        return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY or "not-needed")
    return OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Demo-Korpus (Energie-Themen)
# -----------------------------
def make_demo_corpus(repeat: int = 4) -> str:
    sections = {
        "Solarenergie": [
            "Photovoltaik wandelt Sonnenlicht mittels Halbleitern direkt in Gleichstrom um.",
            "PV-Module speisen über Wechselrichter ins Netz ein; der Ertrag hängt von Einstrahlung und Temperatur ab.",
            "Nachführung kann den Jahresertrag steigern, ist aber mechanisch aufwändiger.",
        ],
        "Windkraft": [
            "Onshore- und Offshore-Anlagen unterscheiden sich u.a. im Kapazitätsfaktor und in den Wartungsanforderungen.",
            "Die Aerodynamik der Rotorblätter bestimmt Anlaufverhalten und Wirkungsgrad.",
            "Moderne Turbinen nutzen Pitch- und Yaw-Regelung zur Leistungsoptimierung."
        ],
        "Batteriespeicher": [
            "Lithium-Ionen-Speicher ermöglichen Peak Shaving und Arbitrage im Strommarkt.",
            "State of Charge und Temperaturmanagement sind zentrale Einflussgrößen auf die Lebensdauer.",
            "Batteriespeicher können sowohl netzdienlich als auch hinter dem Zähler eingesetzt werden."
        ],
        "Netzfrequenz": [
            "In Kontinentaleuropa wird eine Netzfrequenz von 50 Hertz angestrebt.",
            "Frequenzhaltung erfolgt über Primärregelung, Sekundärregelung und Tertiärreserve.",
            "Abweichungen der Frequenz zeigen Ungleichgewichte zwischen Erzeugung und Last an."
        ],
        "Elektromobilität": [
            "AC- und DC-Laden unterscheiden sich in Leistung, Effizienz und Hardwareanforderungen.",
            "Vehicle-to-Grid erlaubt bidirektionales Laden zur Netzdienstleistung.",
            "Ladeinfrastruktur benötigt Lastmanagement, um Netzengpässe zu vermeiden."
        ],
        "Wasserstoff": [
            "PEM-Elektrolyseure reagieren schnell und eignen sich zur flexiblen Erzeugung.",
            "Speicherung erfolgt komprimiert, verflüssigt oder in Materialien wie LOHC.",
            "Brennstoffzellen wandeln chemische Energie mit hohen Teillastwirkungsgraden um."
        ],
        "Regulatorik": [
            "Ausschreibungen und Netzentgelte beeinflussen die Wirtschaftlichkeit erneuerbarer Projekte.",
            "Redispatch 2.0 koordiniert Eingriffe zur Netzstabilität auf Basis von Prognosen.",
            "Fördermechanismen ändern sich über die Zeit und erfordern Monitoring."
        ],
        "Sicherheit": [
            "Systemische Resilienz schützt gegen Ausfälle und Kaskadeneffekte.",
            "Notstromkonzepte und Inselbetrieb sichern kritische Infrastrukturen.",
            "Blackout-Risiken werden durch Schutzsysteme und Regelreserven begrenzt."
        ],
        "Wärmepumpen": [
            "Die Jahresarbeitszahl hängt von Quelltemperatur und Vorlauf ab.",
            "Erdsonden bieten stabile Quelltemperaturen und verbessern die Effizienz.",
            "Hydraulischer Abgleich ist Voraussetzung für niedrige Vorlauftemperaturen."
        ],
        "Messung & Daten": [
            "Smart Meter erfassen Lastgänge typischerweise in 15-Minuten-Intervallen.",
            "Zeitreihenanalyse zeigt Flexibilitätspotenziale und atypische Lastspitzen.",
            "Datenqualität ist entscheidend für Prognosen und Betriebsoptimierung."
        ],
    }
    parts = []
    for _ in range(repeat):
        for title, paras in sections.items():
            parts.append(f"{title}\n" + " ".join(paras))
    return "\n\n".join(parts)

# -----------------------------
# Chunking (wortbasiert mit Satz-Overlap)
# -----------------------------
def split_sentences(text: str) -> List[str]:
    import re
    sents = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [s for s in sents if s]

def chunk_text(text: str,
               max_words: int = MAX_WORDS_PER_CHUNK,
               overlap_sentences: int = OVERLAP_SENTENCES) -> List[Dict]:
    sents = split_sentences(text)
    chunks = []
    i = 0
    while i < len(sents):
        current = []
        n_words = 0
        start_i = i
        while i < len(sents) and (n_words + len(sents[i].split())) <= max_words:
            current.append(sents[i])
            n_words += len(sents[i].split())
            i += 1
        chunk = " ".join(current)
        chunks.append({"id": f"chunk-{len(chunks)+1}",
                       "text": chunk,
                       "start_sentence": start_i})
        # Satz-Overlap
        i = max(i - overlap_sentences, i)
        if i == start_i:  # falls Satz > max_words
            i += 1
    return chunks

# -----------------------------
# Embeddings
# -----------------------------
class Embedder:
    def __init__(self, backend: str = "local"):
        self.backend = backend
        if backend == "local":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise SystemExit("Bitte installiere sentence-transformers: pip install sentence-transformers") from e
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        elif backend == "openai":
            if not OPENAI_API_KEY:
                raise SystemExit("Setze OPENAI_API_KEY für OPENAI-Embeddings.")
            self.client = _openai_client()
        else:
            raise ValueError("backend muss 'local' oder 'openai' sein.")

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "local":
            vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        else:
            resp = self.client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            # L2-Normalisierung für Cosine als Skalarprodukt
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs

# -----------------------------
# Retrieval (Cosine) + fester Threshold
# -----------------------------
def retrieve_top_k_with_threshold(chunks: List[Dict], chunk_vecs: np.ndarray,
                                  query_vec: np.ndarray, k: int = TOP_K,
                                  threshold: float = ACCEPT_THRESHOLD) -> Tuple[List[Tuple[int, float]], float, bool]:
    """
    Gibt Top-k (Index, Score), den genutzten Threshold und eine Bool-Entscheidung zurück.
    Entscheidung = True, wenn top1 >= threshold, sonst False.
    """
    scores = chunk_vecs @ query_vec  # Cosine (bei normalisierten Embeddings)
    idx_sorted = np.argsort(-scores)[:k]
    hits = [(int(i), float(scores[i])) for i in idx_sorted]
    top_ok = len(hits) > 0 and hits[0][1] >= threshold
    return hits, float(threshold), top_ok

# -----------------------------
# Prompt & optionale Generation
# -----------------------------
def build_prompt(selected_chunks: List[Dict], question: str) -> str:
    ctx_lines = []
    for i, ch in enumerate(selected_chunks, start=1):
        ctx_lines.append(f"[Quelle {i}]\n{ch['text']}")
    ctx = "\n\n".join(ctx_lines)
    prompt = f"""Du bist ein präziser Assistent. Beantworte die Frage AUSSCHLIESSLICH anhand des KONTEXTES.
Wenn Informationen fehlen, antworte exakt: "Keine Daten vorhanden."

KONTEXT:
{ctx}

FRAGE:
{question}

ANTWORT (knapp, auf Deutsch):
"""
    return prompt.strip()

def generate_answer(prompt: str) -> str:
    if GEN_BACKEND == "none":
        return "(LLM deaktiviert) Kontext bereitgestellt. Setze GEN_BACKEND=openai oder lmstudio."
    client = _openai_client()
    model_name = OPENAI_CHAT_MODEL if GEN_BACKEND == "openai" else LMSTUDIO_CHAT_MODEL
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# Demo
# -----------------------------
def demo():
    print("== RAG Demo mit Embeddings (fixer Threshold) ==")
    CORPUS_PATH="training_ki_supply_chain.txt"
    if CORPUS_PATH is not None and os.path.isfile(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus = f.read()
        print(f"Lade Korpus aus Datei: {CORPUS_PATH} ({len(corpus.split())} Wörter)\n")
    else:
        corpus = make_demo_corpus(repeat=17)
        print(f"Länge des Korpus: {len(corpus.split())} Wörter\n")

    # 1) Chunking
    chunks = chunk_text(corpus, max_words=MAX_WORDS_PER_CHUNK, overlap_sentences=OVERLAP_SENTENCES)
    print(f"Erzeugte Chunks: {len(chunks)}  (max {MAX_WORDS_PER_CHUNK} Wörter, Overlap {OVERLAP_SENTENCES} Sätze)")

    # 2) Embeddings vorbereiten
    embedder = Embedder(backend=EMBED_BACKEND)
    chunk_vecs = embedder.encode([c["text"] for c in chunks])

    # 3) Beispiel-Fragen
    queries = [
        "Wozu eignen sich Batteriespeicher im Strommarkt?",
        "Was ist KI?",
        # "Was bedeutet Vehicle-to-Grid?",
        "Was ist 1+1?",
        "Was ist 2+2?"
    ]

    for q in queries:
        print("\n" + "-" * 80)
        print("FRAGE:", q)
        q_vec = embedder.encode([q])[0]

        # 4) Retrieval + fester Threshold
        hits, thr, ok = retrieve_top_k_with_threshold(chunks, chunk_vecs, q_vec, k=TOP_K, threshold=ACCEPT_THRESHOLD)
        print(f"Top-{TOP_K} Treffer (Cosine-Similarity, Threshold={thr:.2f}):")
        for rank, (i, score) in enumerate(hits, start=1):
            preview = chunks[i]["text"][:180].replace("\n", " ")
            bar = "█" * int(max(1, round(score * 10)))  # simple Score-Bar
            print(f"{rank:>2}. score={score:.3f} {bar:<10}  |  {chunks[i]['id']}: {preview}...")

        if not ok:
            print("\n--- Entscheidung: NICHT relevant (Top-1 < Threshold) ---")
            print("Antwort: Keine Daten vorhanden.")
            continue

        # 5) Prompt + (6) optionale Generation
        selected = [chunks[i] for i, _ in hits]
        prompt = build_prompt(selected, q)
        print("\n#----- RAG Prompt (gekürzt) -----#")
        preview = prompt[:700] + ("..." if len(prompt) > 700 else "")
        print(preview)

        answer = generate_answer(prompt)
        print("\n#----- Antwort -----#")
        print(answer)

if __name__ == "__main__":
    demo()
