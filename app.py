# ==========================================================
# IMPORTS - Standard Python libraries
# ==========================================================

import os          # Variables d'environnement
import json        # Lecture / écriture JSON
import re          # Expressions régulières
import pandas as pd  # Manipulation et affichage des données


# ==========================================================
# IMPORTS - Document loading
# ==========================================================

from PyPDF2 import PdfReader   # Lecture des fichiers PDF
from docx import Document      # Lecture des fichiers DOCX


# ==========================================================
# IMPORTS - Text chunking
# ==========================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Découpage intelligent des documents en chunks


# ==========================================================
# IMPORTS - OpenAI
# ==========================================================

from openai import OpenAI
# Appels au modèle GPT pour :
# - extraction des triplets
# - extraction des mots-clés
# - réponse GraphRAG


# ==========================================================
# IMPORTS - Neo4j
# ==========================================================

from neo4j import GraphDatabase
# Connexion à la base de données graphe Neo4j


# ==========================================================
# IMPORTS - Graph visualization
# ==========================================================

from pyvis.network import Network
# Construction du graphe interactif

import streamlit.components.v1 as components
# Affichage du graphe PyVis dans Streamlit


# ==========================================================
# IMPORTS - Streamlit
# ==========================================================

import streamlit as st
# Interface utilisateur :
# - upload de fichiers
# - boutons
# - affichage des résultats
# - chatbot GraphRAG

# ==========================================================
# IMPORTS - EVALUATION
# ==========================================================
from rouge_score import rouge_scorer
from bert_score import score

# ==========================================================
# METRICS INITIALIZATION
# ==========================================================

rouge = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

# Configure la page Streamlit
st.set_page_config(
    page_title="GraphRAG DS50",   # Nom de l'onglet navigateur
    layout="wide"                 # Utilise toute la largeur disponible
)

# Titre principal affiché dans l'application
st.title("Interactive GraphRAG DS50")

# Description rapide du projet
st.write(
    "Upload a document, extract triplets, build a knowledge graph, "
    "visualize it, and ask questions."
)

# ==========================================================
# SESSION STATE INITIALIZATION
# ==========================================================

if "triplets" not in st.session_state:
    st.session_state["triplets"] = []

if "failed_outputs" not in st.session_state:
    st.session_state["failed_outputs"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "evaluation_history" not in st.session_state:
    st.session_state["evaluation_history"] = []

# ==========================================================
# TEXT EXTRACTION - MAIN ROUTER
# ==========================================================

def extract_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return load_txt(uploaded_file)

    elif file_name.endswith(".pdf"):
        return load_pdf(uploaded_file)

    elif file_name.endswith(".docx"):
        return load_docx(uploaded_file)

    elif file_name.endswith(".json"):
        return load_json(uploaded_file)

    return ""

# ==========================================================
# TEXT EXTRACTION - TXT
# ==========================================================

def load_txt(uploaded_file):

    return uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

# ==========================================================
# TEXT EXTRACTION - PDF
# ==========================================================

def load_pdf(uploaded_file):

    pdf = PdfReader(uploaded_file)

    pages = []

    for page in pdf.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)

# ==========================================================
# TEXT EXTRACTION - DOCX
# ==========================================================

def load_docx(uploaded_file):

    doc = Document(uploaded_file)

    paragraphs = []

    for paragraph in doc.paragraphs:

        if paragraph.text.strip():

            paragraphs.append(
                paragraph.text
            )

    return "\n".join(paragraphs)

# ==========================================================
# TEXT EXTRACTION - JSON
# ==========================================================

def load_json(uploaded_file):

    data = json.load(uploaded_file)

    if isinstance(data, list):

        return "\n".join(
            [str(item) for item in data]
        )

    elif isinstance(data, dict):

        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )

    return str(data)

# ==========================================================
# CHUNKING FUNCTION
# ==========================================================

def create_chunks(raw_text, chunk_size=500, chunk_overlap=100):
    """
    Découpe le texte brut en chunks avec overlap.
    """

    # Création du splitter LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # Taille maximale d'un chunk
        chunk_overlap=chunk_overlap,    # Partie répétée entre deux chunks
        separators=[
            "\n\n",  # Priorité 1 : paragraphes
            "\n",    # Priorité 2 : lignes
            ".",     # Priorité 3 : phrases
            " ",     # Priorité 4 : mots
            ""       # Dernier recours : caractères
        ]
    )

    # Découpage du texte
    split_chunks = text_splitter.split_text(raw_text)

    # Ajout d'un identifiant à chaque chunk
    chunks = []

    for i, chunk in enumerate(split_chunks):
        chunks.append({
            "text": chunk,
            "chunk_id": f"chunk_{i}"
        })

    return chunks


# =========================
# OpenAI LLM function
# =========================

def get_openai_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY not found in environment variables.")
        return None

    return OpenAI(api_key=api_key)


def llm(prompt, model=None, temperature=0, max_tokens=1500):

    if model is None:
        model = st.session_state.get("selected_model", "gpt-4o-mini")

    client = get_openai_client()

    if client is None:
        return ""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledge graph extraction assistant. "
                        "Extract clean, factual, structured triplets from documents."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"LLM error: {e}")
        return ""

# =========================
# Neo4j connection function
# =========================

def get_neo4j_driver():

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not username or not password:
        st.error("Neo4j credentials are missing in environment variables.")
        return None

    return GraphDatabase.driver(
        uri,
        auth=(username, password)
    )

# =========================
# Neo4j normalization helpers
# =========================

def clean_relation(relation):

    relation = str(relation).strip().upper()
    relation = re.sub(r"[^A-Z0-9_]", "_", relation)
    relation = re.sub(r"_+", "_", relation)
    relation = relation.strip("_")

    if relation == "":
        relation = "RELATED_TO"

    if relation[0].isdigit():
        relation = "REL_" + relation

    return relation
  
# =========================
# Insert triplets into Neo4j
# =========================

def insert_triplets_into_neo4j(triplets):

    driver = get_neo4j_driver()

    if driver is None:
        return 0

    inserted = 0

    with driver.session() as session:

        for t in triplets:

            relation = clean_relation(t["relation"])

            query = f"""
            MERGE (s:Entity {{name: $subject}})
            SET s.type = $subject_type

            MERGE (o:Entity {{name: $object}})
            SET o.type = $object_type

            MERGE (s)-[r:{relation}]->(o)

            SET r.confidence = $confidence,
                r.source = $source
            """

            session.run(
                query,
                subject=t["subject"],
                subject_type=t["subject_type"],
                object=t["object"],
                object_type=t["object_type"],
                confidence=t.get("confidence", 1.0),
                source=t["source"]
            )

            inserted += 1

    driver.close()

    return inserted

# =========================
# Clear Neo4j graph
# =========================

def clear_neo4j_graph():

    driver = get_neo4j_driver()

    if driver is None:
        return False

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    driver.close()

    return True

# ==========================================================
# DELETE NODE FROM NEO4J
# ==========================================================

def delete_node_from_neo4j(node_name):
    """
    Deletes a node and all its relationships from Neo4j.
    """

    driver = get_neo4j_driver()

    if driver is None:
        return False

    query = """
    MATCH (n:Entity {name: $node_name})
    DETACH DELETE n
    """

    with driver.session() as session:
        session.run(query, node_name=node_name)

    driver.close()

    return True

# ==========================================================
# DELETE RELATIONSHIP FROM NEO4J
# ==========================================================

def delete_relationship_from_neo4j(
    subject,
    relation,
    obj
):

    driver = get_neo4j_driver()

    if driver is None:
        return False

    query = f"""
    MATCH (s:Entity {{name: $subject}})
    -[r:{relation}]->
    (o:Entity {{name: $obj}})
    DELETE r
    """

    with driver.session() as session:

        session.run(
            query,
            subject=subject,
            obj=obj
        )

    driver.close()

    return True

# ==========================================================
# MERGE NODES 
# ==========================================================

def merge_nodes_in_neo4j(node_1, node_2, merged_name):

    driver = get_neo4j_driver()

    if driver is None:
        return False

    with driver.session() as session:

        # Création du nœud fusionné
        session.run(
            """
            MERGE (m:Entity {name:$merged_name})
            """,
            merged_name=merged_name
        )

        # Relations sortantes de node_1
        outgoing_1 = session.run(
            """
            MATCH (a:Entity {name:$node})
            -[r]->
            (o:Entity)

            RETURN
                type(r) AS rel_type,
                o.name AS target
            """,
            node=node_1
        )

        for row in outgoing_1:

            session.run(
                f"""
                MATCH (m:Entity {{name:$merged_name}})
                MATCH (o:Entity {{name:$target}})
                MERGE (m)-[:{row['rel_type']}]->(o)
                """,
                merged_name=merged_name,
                target=row["target"]
            )

        # Relations sortantes de node_2
        outgoing_2 = session.run(
            """
            MATCH (a:Entity {name:$node})
            -[r]->
            (o:Entity)

            RETURN
                type(r) AS rel_type,
                o.name AS target
            """,
            node=node_2
        )

        for row in outgoing_2:

            session.run(
                f"""
                MATCH (m:Entity {{name:$merged_name}})
                MATCH (o:Entity {{name:$target}})
                MERGE (m)-[:{row['rel_type']}]->(o)
                """,
                merged_name=merged_name,
                target=row["target"]
            )

        # Relations entrantes de node_1
        incoming_1 = session.run(
            """
            MATCH (s:Entity)
            -[r]->
            (a:Entity {name:$node})

            RETURN
                s.name AS source,
                type(r) AS rel_type
            """,
            node=node_1
        )

        for row in incoming_1:

            session.run(
                f"""
                MATCH (s:Entity {{name:$source}})
                MATCH (m:Entity {{name:$merged_name}})
                MERGE (s)-[:{row['rel_type']}]->(m)
                """,
                source=row["source"],
                merged_name=merged_name
            )

        # Relations entrantes de node_2
        incoming_2 = session.run(
            """
            MATCH (s:Entity)
            -[r]->
            (a:Entity {name:$node})

            RETURN
                s.name AS source,
                type(r) AS rel_type
            """,
            node=node_2
        )

        for row in incoming_2:

            session.run(
                f"""
                MATCH (s:Entity {{name:$source}})
                MATCH (m:Entity {{name:$merged_name}})
                MERGE (s)-[:{row['rel_type']}]->(m)
                """,
                source=row["source"],
                merged_name=merged_name
            )

        # Suppression des anciens nœuds
        session.run(
            """
            MATCH (n:Entity)
            WHERE n.name IN [$node_1, $node_2]
            AND n.name <> $merged_name
            DETACH DELETE n
            """,
            node_1=node_1,
            node_2=node_2,
            merged_name=merged_name
        )

    driver.close()

    return True

# ==========================================================
# ADD RELATIONSHIP TO NEO4J
# ==========================================================

def add_relationship_to_neo4j(
    subject,
    relation,
    obj
):

    driver = get_neo4j_driver()

    if driver is None:
        return False

    query = f"""
    MERGE (s:Entity {{name:$subject}})
    MERGE (o:Entity {{name:$obj}})
    MERGE (s)-[:{relation}]->(o)
    """

    with driver.session() as session:

        session.run(
            query,
            subject=subject,
            obj=obj
        )

    driver.close()

    return True

# ==========================================================
# EDIT RELATIONSHIP 
# ==========================================================

def edit_relationship_type_in_neo4j(subject, old_relation, new_relation, obj):

    driver = get_neo4j_driver()

    if driver is None:
        return False

    old_relation = clean_relation(old_relation)
    new_relation = clean_relation(new_relation)

    query = f"""
    MATCH (s:Entity {{name:$subject}})-[r:{old_relation}]->(o:Entity {{name:$obj}})
    MERGE (s)-[new_r:{new_relation}]->(o)
    SET new_r.confidence = r.confidence,
        new_r.source = r.source
    DELETE r
    RETURN s.name AS subject, type(new_r) AS relation, o.name AS object
    """

    with driver.session() as session:
        result = session.run(
            query,
            subject=subject,
            obj=obj
        )

        records = list(result)

    driver.close()

    return len(records) > 0

# =========================
# Fetch graph from Neo4j
# =========================

def fetch_graph_from_neo4j(limit=600):

    driver = get_neo4j_driver()

    if driver is None:
        return []

    query = """
    MATCH (s:Entity)-[r]->(o:Entity)
    RETURN
        s.name AS subject,
        s.type AS subject_type,
        type(r) AS relation,
        o.name AS object,
        o.type AS object_type,
        r.confidence AS confidence,
        r.source AS source
    LIMIT $limit
    """

    with driver.session() as session:
        result = session.run(query, limit=limit)

        graph_data = []

        for record in result:
            graph_data.append(dict(record))

    driver.close()

    return graph_data

# ==========================================================
# FETCH LOCAL SUBGRAPH FROM NEO4J
# ==========================================================

def fetch_local_subgraph_from_neo4j(node_name, limit=100):
    """
    Fetches relationships directly connected to a specific node.
    Useful to visualize recent human edits such as merge, delete, or manual additions.
    """

    driver = get_neo4j_driver()

    if driver is None:
        return []

    query = """
    MATCH (s:Entity)-[r]-(o:Entity)
    WHERE s.name = $node_name OR o.name = $node_name
    RETURN
        s.name AS subject,
        s.type AS subject_type,
        type(r) AS relation,
        o.name AS object,
        o.type AS object_type,
        r.confidence AS confidence,
        r.source AS source
    LIMIT $limit
    """

    with driver.session() as session:

        result = session.run(
            query,
            node_name=node_name,
            limit=limit
        )

        graph_data = [dict(record) for record in result]

    driver.close()

    return graph_data

# =========================
# Visualize graph with PyVis
# =========================

def visualize_graph(graph_data):

    net = Network(
        height="600px",
        width="100%",
        directed=True
    )

    for item in graph_data:

        subject = item["subject"]
        obj = item["object"]
        relation = item["relation"]

        subject_type = item.get("subject_type", "Entity")
        object_type = item.get("object_type", "Entity")

        net.add_node(
            subject,
            label=subject,
            title=f"Type: {subject_type}"
        )

        net.add_node(
            obj,
            label=obj,
            title=f"Type: {object_type}"
        )

        net.add_edge(
            subject,
            obj,
            label=relation,
            title=f"Relation: {relation}"
        )

    net.save_graph("graph.html")

    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()

    components.html(
        html,
        height=650,
        scrolling=True
    )
# =========================
# Keyword extraction for GraphRAG
# =========================

def extract_keywords(question, max_keywords=10):

    prompt = f"""
You are a keyword extraction assistant for a GraphRAG system.

Extract the most useful search keywords from the user question.

Return ONLY a comma-separated list.

Rules:
- Keep named entities.
- Keep important concepts.
- Keep possible synonyms.
- Keep action verbs.
- Remove useless words.
- Do not write explanations.
- Return between 3 and {max_keywords} keywords.

Question:
{question}

Keywords:
"""

    output = llm(prompt)

    output = output.replace("\n", ",")

    keywords = [
        k.strip().lower()
        for k in output.split(",")
        if k.strip() and len(k.strip()) > 2
    ]

    keywords = list(dict.fromkeys(keywords))

    return keywords[:max_keywords]

# ==========================================================
# GRAPHRAG QUESTION ANSWERING
# ==========================================================

def ask_graph(question, top_k=40):
    """
    Answers a user question using the Neo4j knowledge graph.

    Steps:
    1. Extract useful keywords from the question.
    2. Add original question words as fallback keywords.
    3. Retrieve matching graph triplets from Neo4j.
    4. Retrieve neighboring nodes for multi-hop reasoning.
    5. Build graph context.
    6. Ask the LLM to answer using only this context.
    """

    # Extract LLM-generated keywords
    keywords = extract_keywords(question)

    # Fallback: also keep useful words from the original question
    question_words = [
        w.lower().strip(" ?.,;:!()[]{}\"'")
        for w in question.split()
        if len(w.strip(" ?.,;:!()[]{}\"'")) > 2
    ]

    # Merge keywords and remove duplicates
    keywords = list(dict.fromkeys(keywords + question_words))

    if not keywords:
        return "The question does not contain enough useful keywords."

    driver = get_neo4j_driver()

    if driver is None:
        return "Neo4j connection error."

    with driver.session() as session:

        result = session.run("""
        MATCH (s:Entity)-[r]->(o:Entity)

        WHERE ANY(k IN $keywords WHERE
            toLower(s.name) CONTAINS k
            OR toLower(o.name) CONTAINS k
            OR toLower(type(r)) CONTAINS k
            OR toLower(s.type) CONTAINS k
            OR toLower(o.type) CONTAINS k
        )

        OPTIONAL MATCH (o)-[r2]->(x:Entity)

        RETURN
            s.name AS subject,
            s.type AS subject_type,
            type(r) AS relation,
            o.name AS object,
            o.type AS object_type,
            r.confidence AS confidence,
            r.source AS source,

            collect({
                relation: type(r2),
                object: x.name,
                object_type: x.type
            }) AS neighbors,

            size([
                k IN $keywords WHERE
                toLower(s.name) CONTAINS k
                OR toLower(o.name) CONTAINS k
                OR toLower(type(r)) CONTAINS k
                OR toLower(s.type) CONTAINS k
                OR toLower(o.type) CONTAINS k
            ]) AS keyword_score

        ORDER BY keyword_score DESC, confidence DESC
        LIMIT 80
        """, keywords=keywords)

        records = list(result)

    driver.close()

    triples = []

    for record in records:

        triples.append({
            "subject": record["subject"],
            "subject_type": record["subject_type"],
            "relation": record["relation"],
            "object": record["object"],
            "object_type": record["object_type"],
            "confidence": record["confidence"] or 0.7,
            "source": record["source"],
            "score": record["keyword_score"],
            "neighbors": record["neighbors"],
        })

    if not triples:
        return "The graph does not contain enough information."

    selected_triples = triples[:top_k]

    context_lines = []

    for t in selected_triples:

        context_lines.append(
            f"{t['subject']} ({t['subject_type']}) "
            f"--{t['relation']}--> "
            f"{t['object']} ({t['object_type']}) "
            f"[confidence={t['confidence']}, source={t['source']}]"
        )

        for n in t.get("neighbors", []):

            if n["relation"] and n["object"]:

                context_lines.append(
                    f"{t['object']} ({t['object_type']}) "
                    f"--{n['relation']}--> "
                    f"{n['object']} ({n['object_type']})"
                )

    context = "\n".join(context_lines)

    prompt = f"""
You are a GraphRAG assistant.

Answer the question using ONLY the factual graph context below.

Rules:
- Do not invent information.
- Combine related graph facts when the answer requires multi-hop reasoning.
- If the graph does not contain the answer, say:
  "The graph does not contain enough information."
- Give a clear and concise answer.
- Use the most relevant graph facts.
- Do not mention internal scores unless useful.

Question:
{question}

Graph context:
{context}

Answer:
"""

    answer = llm(prompt)

    return answer

# ==========================================================
# JSON EXTRACTION HELPER
# ==========================================================

def extract_json_array(output):
    """
    Extracts a valid JSON array from the LLM output.

    The LLM should return only JSON, but sometimes it may add
    extra text before or after the JSON array. This function tries
    to recover the JSON safely.
    """

    # First attempt: direct JSON parsing
    try:
        data = json.loads(output)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        pass

    # Second attempt: extract the first JSON array found in the text
    match = re.search(r"\[.*\]", output, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(0))

            if isinstance(data, list):
                return data

            return []

        except Exception:
            return []

    # If no valid JSON array is found
    return []


# ==========================================================
# TRIPLET EXTRACTION FUNCTION
# ==========================================================

def extract_triplets_from_chunks(chunks, max_chunks=None):
    """
    Extracts knowledge graph triplets from document chunks using the LLM.

    Input:
    - chunks: list of text chunks
    - max_chunks: number of chunks to process

    Output:
    - triplets_all: valid extracted triplets
    - failed_outputs: LLM outputs that could not be parsed
    """

    triplets_all = []
    failed_outputs = []

    # If max_chunks is not specified, process all chunks
    if max_chunks is None:
        max_chunks = len(chunks)

    # Avoid processing more chunks than available
    max_chunks = min(max_chunks, len(chunks))

    progress = st.progress(0)

    for i, chunk in enumerate(chunks[:max_chunks]):

        text = chunk["text"]

        prompt = f"""
You are an information extraction system for a GraphRAG pipeline.

Your task is to extract high-quality knowledge graph triplets from the text.

Return ONLY a valid JSON array.

Each item must follow this format:
[
  {{
    "subject": "...",
    "relation": "...",
    "object": "...",
    "subject_type": "...",
    "object_type": "...",
    "confidence": 0.0
  }}
]

Rules:
- Extract only facts explicitly stated in the text.
- Do not invent information.
- Extract as many useful factual relations as possible.
- Prefer specific relations over vague ones.
- Avoid vague relations such as "is", "has", "related to", "mentions".
- Use short and normalized entity names.
- Do not use pronouns as entities.
- Entity types can be: Person, Organization, Place, Product, Event, Concept, Date, Other.
- Confidence must be a number between 0 and 1.
- Return only valid JSON.
- Do not add explanations.

Text:
{text}

JSON:
"""

        output = llm(prompt)

        data = extract_json_array(output)

        if not data:
            failed_outputs.append({
                "source": chunk["chunk_id"],
                "output": output
            })
            continue

        for t in data:

            subject = str(t.get("subject", "")).strip()
            relation = str(t.get("relation", "")).strip()
            obj = str(t.get("object", "")).strip()

            # Minimal validation
            if len(subject) < 2 or len(relation) < 2 or len(obj) < 2:
                continue

            if subject.lower() == obj.lower():
                continue

            triplets_all.append({
                "subject": subject,
                "relation": relation,
                "object": obj,
                "subject_type": str(t.get("subject_type", "Other")).strip(),
                "object_type": str(t.get("object_type", "Other")).strip(),
                "confidence": float(t.get("confidence", 0.7)),
                "source": chunk["chunk_id"]
            })

        progress.progress((i + 1) / max_chunks)

    return triplets_all, failed_outputs

# ==========================================================
# RENAME NODE IN NEO4J
# ==========================================================

def rename_node_in_neo4j(
    old_name,
    new_name
):

    driver = get_neo4j_driver()

    if driver is None:
        return False

    query = """
    MATCH (n:Entity {name:$old_name})
    SET n.name = $new_name
    """

    with driver.session() as session:

        session.run(
            query,
            old_name=old_name,
            new_name=new_name
        )

    driver.close()

    return True


# =========================
# Sidebar parameters
# =========================

st.sidebar.header("Pipeline settings")

st.sidebar.header("LLM Settings")

selected_model = st.sidebar.selectbox(
    "Model",
    [
        "gpt-4o-mini",
        "gpt-4o"
    ],
    index=0
)

st.session_state["selected_model"] = selected_model

st.sidebar.success(
    f"Active model: {st.session_state['selected_model']}"
)

chunk_size = st.sidebar.number_input(
    "Chunk size",
    min_value=200,
    max_value=2000,
    value=500,
    step=100
)

chunk_overlap = st.sidebar.number_input(
    "Chunk overlap",
    min_value=0,
    max_value=500,
    value=100,
    step=50
)


# =========================
# Upload document
# =========================

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx", "json"]
)


# ==========================================================
# MAIN PIPELINE
# ==========================================================

if uploaded_file is not None:

    # ------------------------------------------------------
    # 1. Detect new uploaded document
    # ------------------------------------------------------

    current_file_name = uploaded_file.name

    if st.session_state.get("current_file_name") != current_file_name:

        st.session_state["current_file_name"] = current_file_name
        st.session_state.pop("triplets", None)
        st.session_state.pop("failed_outputs", None)
    # --------------------------------------------------
    # Session State Initialization
    # --------------------------------------------------

    if "triplets" not in st.session_state:
        st.session_state["triplets"] = []

    if "failed_outputs" not in st.session_state:
        st.session_state["failed_outputs"] = []

    # ------------------------------------------------------
    # 2. Display file information
    # ------------------------------------------------------

    st.success(f"File uploaded: {uploaded_file.name}")

    st.write("### File information")

    st.json({
        "Filename": uploaded_file.name,
        "File type": uploaded_file.type,
        "File size (KB)": round(uploaded_file.size / 1024, 2)
    })

    # ------------------------------------------------------
    # 3. Extract raw text
    # ------------------------------------------------------

    raw_text = extract_text(uploaded_file)

    st.write("### Extracted text preview")

    st.text_area(
        "Raw text",
        raw_text[:5000],
        height=300
    )

    st.write(f"Text length: {len(raw_text)} characters")

    # ------------------------------------------------------
    # 4. Continue only if text was extracted
    # ------------------------------------------------------

    if raw_text.strip():

        # --------------------------------------------------
        # 5. Create chunks
        # --------------------------------------------------

        chunks = create_chunks(
            raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        st.write("### Chunking results")
        st.write(f"Number of chunks: {len(chunks)}")

        if chunks:

            st.text_area(
                "First chunk preview",
                chunks[0]["text"],
                height=250
            )

            with st.expander("Show first 5 chunks"):

                for chunk in chunks[:5]:

                    st.write(f"**{chunk['chunk_id']}**")
                    st.write(chunk["text"])
                    st.divider()

            # --------------------------------------------------
            # 6. Triplet extraction
            # --------------------------------------------------

            st.write("### Triplet extraction")

            max_chunks = st.number_input(
                "Max chunks to process",
                min_value=1,
                max_value=len(chunks),
                value=len(chunks),
                step=1
            )

            if st.button("Extract Triplets"):

                with st.spinner("Extracting triplets with OpenAI..."):

                    triplets, failed_outputs = extract_triplets_from_chunks(
                        chunks,
                        max_chunks=max_chunks
                    )

                st.session_state["triplets"] = triplets
                st.session_state["failed_outputs"] = failed_outputs

                st.success(f"Extracted triplets: {len(triplets)}")
                st.warning(f"Failed outputs: {len(failed_outputs)}")

            # --------------------------------------------------
            # 7. Human-in-the-loop triplet editor
            # --------------------------------------------------

            if len(st.session_state["triplets"]) > 0:

                st.write("### Human-in-the-loop triplet editor")

                st.write(
                    "You can manually edit, add, or delete extracted triplets "
                    "before inserting them into Neo4j."
                )

                triplets_df = pd.DataFrame(
                    st.session_state["triplets"]
                )

                edited_triplets_df = st.data_editor(
                    triplets_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="triplet_editor"
                )


                st.success(
                    f"Current triplets after manual editing: "
                    f"{len(st.session_state['triplets'])}"
                )
                # --------------------------------------------------
                # Save edited triplets
                # --------------------------------------------------

                if st.button("Save Edited Triplets"):

                    st.session_state["triplets"] = edited_triplets_df.to_dict(
                        orient="records"
                    )

                    st.success(
                        f"Saved {len(st.session_state['triplets'])} edited triplets."
                    )

                # ==========================================================
                # EXPORT TRIPLETS TO CSV
                # ==========================================================

                if st.session_state["triplets"]:

                    triplets_export_df = pd.DataFrame(
                        st.session_state["triplets"]
                    )

                    triplets_csv = triplets_export_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Triplets as CSV",
                        data=triplets_csv,
                        file_name="graphrag_triplets.csv",
                        mime="text/csv"
                    )
            else:
                st.info(
                    "Extract triplets first to enable Human-in-the-Loop editing."
                )
    else:

        st.warning("No text could be extracted from this document.")

else:

    st.info("Please upload a TXT, PDF, DOCX, or JSON document to start.")

# ==========================================================
# MANUAL TRIPLET CREATION
# ==========================================================

st.write("### Add a triplet manually")

with st.form("add_triplet_form"):

    subject = st.text_input("Subject")

    relation = st.text_input("Relation")

    obj = st.text_input("Object")

    subject_type = st.selectbox(
        "Subject type",
        [
            "Person",
            "Organization",
            "Place",
            "Product",
            "Event",
            "Concept",
            "Date",
            "Other"
        ]
    )

    object_type = st.selectbox(
        "Object type",
        [
            "Person",
            "Organization",
            "Place",
            "Product",
            "Event",
            "Concept",
            "Date",
            "Other"
        ]
    )

    submitted = st.form_submit_button(
        "Add Triplet"
    )

    if submitted:

        if subject.strip() and relation.strip() and obj.strip():

            st.session_state["triplets"].append({

                "subject": subject.strip(),

                "relation": relation.strip(),

                "object": obj.strip(),

                "subject_type": subject_type,

                "object_type": object_type,

                "source": "manual"

            })

            st.success(
                "Manual triplet added."
            )

        else:

            st.warning(
                "Subject, relation and object are required."
            )

# Preview current triplets

st.write(
    f"Current triplets available: "
    f"{len(st.session_state['triplets'])}"
)

# ==========================================================
# NEO4J CONTROLS
# ==========================================================

st.write("### Neo4j controls")

if st.button("Clear Neo4j Graph"):

    with st.spinner("Deleting graph..."):

        clear_neo4j_graph()

    st.success("Neo4j graph cleared.")

if st.button("Insert current triplets into Neo4j"):

    if "triplets" in st.session_state:

        with st.spinner("Inserting triplets into Neo4j..."):

            inserted_count = insert_triplets_into_neo4j(
                st.session_state["triplets"]
            )

        st.success(
            f"Inserted triplets into Neo4j: {inserted_count}"
        )

    else:

        st.warning("No triplets available. Please extract triplets first.")

# ==========================================================
# DELETE NODE
# ==========================================================
st.write("### Delete Node")

node_name = st.text_input(
    "Node name to delete"
)

if st.button("Delete Node"):

    if node_name.strip():

        delete_node_from_neo4j(
            node_name.strip()
        )

        st.success(
            f"Node '{node_name}' deleted."
        )

    else:

        st.warning(
            "Please enter a node name."
        )

# ==========================================================
# ADD RELATIONSHIP
# ==========================================================

st.write("### Add Relationship")

add_subject = st.text_input(
    "Relationship Subject",
    key="add_rel_subject"
)

add_relation = st.text_input(
    "Relationship Type",
    key="add_rel_type"
)

add_object = st.text_input(
    "Relationship Object",
    key="add_rel_object"
)

if st.button("Add Relationship"):

    if (
        add_subject.strip()
        and add_relation.strip()
        and add_object.strip()
    ):

        add_relationship_to_neo4j(
            add_subject.strip(),
            add_relation.strip(),
            add_object.strip()
        )

        st.success(
            "Relationship added."
        )

    else:

        st.warning(
            "Please fill all fields."
        )

# ==========================================================
# EDIT RELATIONSHIP TYPE
# ==========================================================

st.write("### Edit Relationship Type")

edit_rel_subject = st.text_input(
    "Edit relationship subject",
    key="edit_rel_subject"
)

edit_old_relation = st.text_input(
    "Current relationship type",
    key="edit_old_relation"
)

edit_new_relation = st.text_input(
    "New relationship type",
    key="edit_new_relation"
)

edit_rel_object = st.text_input(
    "Edit relationship object",
    key="edit_rel_object"
)

if st.button("Edit Relationship Type"):

    if (
        edit_rel_subject.strip()
        and edit_old_relation.strip()
        and edit_new_relation.strip()
        and edit_rel_object.strip()
    ):

        success = edit_relationship_type_in_neo4j(
            edit_rel_subject.strip(),
            edit_old_relation.strip(),
            edit_new_relation.strip(),
            edit_rel_object.strip()
        )

        if success:

            st.success("Relationship type updated.")

        else:

            st.warning("No matching relationship found.")

    else:

        st.warning("Please fill all fields.")

# ==========================================================
# DELETE RELATIONSHIP
# ==========================================================

st.write("### Delete Relationship")

rel_subject = st.text_input(
    "Relationship Subject",
    key="delete_rel_subject"
)

rel_type = st.text_input(
    "Relationship Type",
    key="delete_rel_type"
)

rel_object = st.text_input(
    "Relationship Object",
    key="delete_rel_object"
)

if st.button("Delete Relationship"):

    if (
        rel_subject.strip()
        and rel_type.strip()
        and rel_object.strip()
    ):

        delete_relationship_from_neo4j(
            rel_subject.strip(),
            rel_type.strip(),
            rel_object.strip()
        )

        st.success(
            "Relationship deleted."
        )

    else:

        st.warning(
            "Please fill all fields."
        )

# ==========================================================
# RENAME NODE
# ==========================================================

st.write("### Rename Node")

old_node_name = st.text_input(
    "Current node name",
    key="rename_old_node"
)

new_node_name = st.text_input(
    "New node name",
    key="rename_new_node"
)

if st.button("Rename Node"):

    if (
        old_node_name.strip()
        and new_node_name.strip()
    ):

        rename_node_in_neo4j(
            old_node_name.strip(),
            new_node_name.strip()
        )

        st.success(
            f"Node renamed from '{old_node_name}' to '{new_node_name}'."
        )

    else:

        st.warning(
            "Please fill all fields."
        )

# ==========================================================
# MERGE NODES
# ==========================================================

st.write("### Merge Nodes")

merge_node_1 = st.text_input(
    "Node 1"
)

merge_node_2 = st.text_input(
    "Node 2"
)

merged_name = st.text_input(
    "Merged node name"
)

if st.button("Merge Nodes"):

    if (
        merge_node_1.strip()
        and merge_node_2.strip()
        and merged_name.strip()
    ):

        merge_nodes_in_neo4j(
            merge_node_1.strip(),
            merge_node_2.strip(),
            merged_name.strip()
        )

        st.success(
            f"Nodes merged into '{merged_name}'."
        )

    else:

        st.warning(
            "Please fill all fields."
        )

# ==========================================================
# GRAPH VISUALIZATION
# ==========================================================

st.write("### Graph visualization")

if st.button("Visualize Graph"):

    graph_data = fetch_graph_from_neo4j()

    st.success(
        f"Loaded {len(graph_data)} relationships"
    )

    visualize_graph(graph_data)

# ==========================================================
# LOCAL GRAPH VISUALIZATION
# ==========================================================

st.write("### Visualize Local Subgraph")

local_node_name = st.text_input(
    "Node name for local visualization"
)

if st.button("Visualize Local Subgraph"):

    if local_node_name.strip():

        local_graph_data = fetch_local_subgraph_from_neo4j(
            local_node_name.strip()
        )

        st.success(
            f"Loaded {len(local_graph_data)} local relationships"
        )

        if local_graph_data:

            visualize_graph(local_graph_data)

        else:

            st.warning(
                "No local relationships found for this node."
            )

    else:

        st.warning(
            "Please enter a node name."
        )

# ==========================================================
# GRAPHRAG CHATBOT WITH HISTORY
# ==========================================================

st.header("GraphRAG Chatbot")

# Initialise chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User question input
question = st.text_input(
    "Ask a question about the graph"
)

# Ask button
if st.button("Ask GraphRAG"):

    if question.strip():

        with st.spinner("Searching Neo4j and generating answer..."):

            answer = ask_graph(question)

        # Store question and answer in history
        st.session_state["chat_history"].append({
            "question": question,
            "answer": answer
        })

    else:

        st.warning("Please enter a question first.")

# Display chat history
if st.session_state["chat_history"]:

    st.subheader("Chat history")

    for i, item in enumerate(
        reversed(st.session_state["chat_history"]),
        1
    ):

        st.markdown(f"**Question {i}:** {item['question']}")
        st.markdown(f"**Answer:** {item['answer']}")
        st.divider()

# Clear chat history
if st.button("Clear Chat History"):

    st.session_state["chat_history"] = []
    st.success("Chat history cleared.")

# ==========================================================
# EVALUATION
# ==========================================================

st.write("## Evaluation")

eval_question = st.text_area(
    "Question",
    key="eval_question"
)

reference_answer = st.text_area(
    "Reference Answer",
    key="reference_answer"
)

if st.button("Compute Metrics"):

    if eval_question.strip() and reference_answer.strip():

        generated_answer = ask_graph(
            eval_question.strip()
        )

        st.write("### Generated Answer")

        st.write(generated_answer)

        rouge_scores = rouge.score(
            reference_answer,
            generated_answer
        )

        rouge1 = rouge_scores["rouge1"].fmeasure
        rouge2 = rouge_scores["rouge2"].fmeasure
        rougeL = rouge_scores["rougeL"].fmeasure

        _, _, F1 = score(
            [generated_answer],
            [reference_answer],
            lang="en",
            verbose=False
        )

        bert_f1 = F1.mean().item()

        st.session_state["evaluation_history"].append({
                "Question": eval_question,
                "ROUGE-1": round(rouge1, 4),
                "ROUGE-2": round(rouge2, 4),
                "ROUGE-L": round(rougeL, 4),
                "BERT F1": round(bert_f1, 4)
            })

# ==========================================================
# EVALUATION HISTORY
# ==========================================================

if st.session_state["evaluation_history"]:

    st.write("### Evaluation History")

    history_df = pd.DataFrame(
        st.session_state["evaluation_history"]
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

# ==========================================================
# EXPORT EVALUATION HISTORY TO TXT
# ==========================================================

if st.session_state["evaluation_history"]:

    export_text = "GraphRAG Evaluation Report\n"
    export_text += "=" * 40 + "\n\n"

    for i, item in enumerate(
        st.session_state["evaluation_history"],
        1
    ):

        export_text += f"Evaluation {i}\n"
        export_text += "-" * 40 + "\n"
        export_text += f"Question: {item.get('Question', '')}\n"
        export_text += f"ROUGE-1: {item.get('ROUGE-1', '')}\n"
        export_text += f"ROUGE-2: {item.get('ROUGE-2', '')}\n"
        export_text += f"ROUGE-L: {item.get('ROUGE-L', '')}\n"
        export_text += f"BERT F1: {item.get('BERT F1', '')}\n\n"

    st.download_button(
        label="Download Evaluation Report as TXT",
        data=export_text.encode("utf-8"),
        file_name="graphrag_evaluation_report.txt",
        mime="text/plain"
    )