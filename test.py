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
<title>Hybrid AI Chatbot</title>

<style>

body{
background:#0f172a;
color:white;
font-family:Arial;
text-align:center;
}

#chat{
width:60%;
margin:auto;
height:400px;
overflow:auto;
border:1px solid gray;
padding:15px;
background:#1e293b;
border-radius:10px;
}

input{
width:50%;
padding:10px;
margin-top:10px;
}

button{
padding:10px;
background:#38bdf8;
border:none;
cursor:pointer;
}

</style>

</head>

<body>

<h2>PDF + Internet AI Chatbot</h2>

<div id="chat"></div>

<br>

<input id="question" placeholder="Ask anything">
<button onclick="ask()">Send</button>

<script>

function ask(){

let q = document.getElementById("question").value;

fetch("/ask",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({question:q})
})

.then(res=>res.json())

.then(data=>{

let chat=document.getElementById("chat");

chat.innerHTML += "<p><b>You:</b> "+q+"</p>";

chat.innerHTML += "<p><b>AI:</b> "+data.answer+"<br><i>Source: "+data.source+"</i></p>";

document.getElementById("question").value="";

chat.scrollTop = chat.scrollHeight;

})

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