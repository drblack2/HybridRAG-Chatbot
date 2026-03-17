import os
from flask import Flask, request, jsonify, render_template_string

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM as Ollama

from duckduckgo_search import DDGS

app = Flask(__name__)

# -----------------------------
# Load PDFs
# -----------------------------

PDF_FOLDER = "data"

documents = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
        documents.extend(loader.load())

print("PDFs loaded")

# -----------------------------
# Split Documents
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

docs = splitter.split_documents(documents)

print("Documents split")

# -----------------------------
# Embedding Model
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings loaded")

# -----------------------------
# Vector Database
# -----------------------------

vector_db = FAISS.from_documents(docs, embeddings)

print("Vector database ready")

# -----------------------------
# Local LLM
# -----------------------------

llm = Ollama(
    model="tinyllama",
    base_url="http://localhost:11434",
    temperature=0.3
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k":3})
)

print("TinyLlama Chatbot ready")

# -----------------------------
# HTML UI
# -----------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>HybridRAG Chatbot</title>

<style>
body {
    margin: 0;
    font-family: 'Segoe UI', sans-serif;
    background: #020617;
    color: white;
}

/* Header */
.header {
    text-align: center;
    padding: 15px;
    font-size: 22px;
    font-weight: bold;
    background: #020617;
    border-bottom: 1px solid #1e293b;
}

/* Chat container */
#chat {
    width: 60%;
    margin: 20px auto;
    height: 65vh;
    overflow-y: auto;
    padding: 20px;
    background: #020617;
    border-radius: 12px;
}

/* Message bubbles */
.msg {
    margin: 10px 0;
    padding: 12px;
    border-radius: 10px;
    max-width: 75%;
    line-height: 1.5;
}

.user {
    background: #2563eb;
    margin-left: auto;
    text-align: right;
}

.bot {
    background: #1e293b;
}

/* Input area */
.input-area {
    width: 60%;
    margin: auto;
    display: flex;
    gap: 10px;
}

input {
    flex: 1;
    padding: 12px;
    border-radius: 8px;
    border: none;
    background: #1e293b;
    color: white;
    outline: none;
}

button {
    padding: 12px 18px;
    border-radius: 8px;
    border: none;
    background: #38bdf8;
    font-weight: bold;
    cursor: pointer;
    transition: 0.2s;
}

button:hover {
    background: #0ea5e9;
}

/* Scrollbar */
#chat::-webkit-scrollbar {
    width: 6px;
}
#chat::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}
</style>

</head>

<body>

<div class="header">🤖 HybridRAG Chatbot</div>

<div id="chat"></div>

<div class="input-area">
    <input id="question" placeholder="Ask anything..." onkeydown="handleKey(event)">
    <button onclick="ask()">Send</button>
</div>

<script>

function addMessage(text, type) {
    let chat = document.getElementById("chat");

    let div = document.createElement("div");
    div.className = "msg " + type;
    div.innerHTML = text;

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function ask() {

    let q = document.getElementById("question").value;

    if (!q) return;

    addMessage("<b>You:</b> " + q, "user");

    fetch("/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: q })
    })
    .then(res => res.json())
    .then(data => {

        addMessage(
            "<b>AI:</b> " + data.answer + "<br><small>(" + data.source + ")</small>",
            "bot"
        );

    });

    document.getElementById("question").value = "";
}

function handleKey(e) {
    if (e.key === "Enter") {
        ask();
    }
}

</script>

</body>
</html>
"""

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        question = data.get("question")

        if not question:
            return jsonify({"answer": "Please ask a question."})

        # --------------------------------
        # Search inside PDF vector database
        # --------------------------------

        docs_and_scores = vector_db.similarity_search_with_score(question, k=3)

        best_score = docs_and_scores[0][1]

        # Lower score means better match

        if best_score < 0.5:

            result = qa.invoke({"query": question})

            answer = result["result"]

            source = "PDF Knowledge Base"

        else:

            # --------------------------------
            # Internet Search
            # --------------------------------

            web_text = ""

            with DDGS() as ddgs:

                results = ddgs.text(question, max_results=3)

                for r in results:
                    web_text += r["body"] + "\n"

            prompt = f"""

Use the following internet information to answer the question.

Information:
{web_text}

Question:
{question}

Answer clearly.
"""

            answer = llm.invoke(prompt)

            source = "Internet Search"

        return jsonify({
            "answer": answer,
            "source": source
        })

    except Exception as e:

        return jsonify({"answer": str(e), "source":"error"}), 500


# -----------------------------
# Run Server
# -----------------------------

if __name__ == "__main__":
    app.run(port=5000, debug=True)