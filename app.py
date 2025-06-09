from flask import Flask, request, session, redirect, url_for, render_template
from flask_bcrypt import Bcrypt
import sqlite3
from datetime import datetime
from rag_pipeline import initialize_pipeline
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from auth import init_auth  # Import the authentication module

# Load environment variables
load_dotenv()
app = Flask(__name__)
secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    raise ValueError("SECRET_KEY must be set in the .env file")
app.secret_key = secret_key
bcrypt = Bcrypt(app)

# Initialize authentication routes
init_auth(app)

# Initialize RAG pipeline
pdf_path = "C:\\Users\\user\\Desktop\\project\\Budget_in_Brief.pdf"
print("Loading PDF from:", pdf_path)
rag_chain, error = initialize_pipeline(pdf_path)
if error:
    raise RuntimeError(f"Failed to initialize RAG pipeline: {error}")

# Initialize follow-up question generator
groq_llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
follow_up_prompt = PromptTemplate(
    input_variables=["answer"],
    template="""
    You are a business insights assistant specializing in fiscal and economic analysis. Based on the following answer, suggest exactly 3 relevant follow-up questions for a business user analyzing the Federal Budget 2024-25. Format the questions as a numbered list starting with 1., 2., and 3., with each question on a new line. Do not include any additional text before or after the numbered list.
    Answer: {answer}
    1. 
    2. 
    3. 
    """
)
follow_up_chain = follow_up_prompt | groq_llm

# Initialize chat memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Update RAG chain with memory
prompt_template = """
You are a business insights assistant specializing in fiscal and economic analysis for a {role}. Use the provided context from the Federal Budget 2024-25 and chat history to answer the question concisely and accurately, focusing on financial metrics, policies, or priorities.
Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template).partial(role=lambda: session.get("role", "user"))
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=groq_llm,
    retriever=rag_chain.retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Database setup (for history and feedback)
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        question TEXT,
        answer TEXT,
        timestamp TEXT,
        starred INTEGER
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        question TEXT,
        answer TEXT,
        rating INTEGER
    )""")
    conn.commit()
    conn.close()

init_db()

# Route handlers
@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response to suppress the error

@app.route("/chat")
def chat():
    if "email" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html", greeting=f"Welcome, {session['name']}!")

@app.route("/ask", methods=["POST"])
def ask():
    if "email" not in session:
        return {"error": "Unauthorized"}, 401
    question = request.json["question"]
    result = conversational_chain({"question": question})
    return {"answer": result["answer"]}

@app.route("/suggest_follow_ups", methods=["POST"])
def suggest_follow_ups():
    answer = request.json["answer"]
    print(f"Received answer for follow-up: {answer}")
    try:
        follow_ups = follow_up_chain.invoke({"answer": answer})
        print(f"Raw follow-ups from Groq: {follow_ups}")
        # Split by lines and filter for non-empty lines
        lines = [line.strip() for line in follow_ups.split("\n") if line.strip()]
        # Extract questions (expecting 1., 2., 3. format)
        processed_follow_ups = []
        current_question = ""
        for line in lines:
            if line.startswith(("1.", "2.", "3.")):
                if current_question:
                    processed_follow_ups.append(current_question)
                current_question = line
            elif current_question:
                current_question += " " + line
        if current_question:
            processed_follow_ups.append(current_question)
        # Fallback: If no numbered questions found, take any non-empty lines
        if not processed_follow_ups:
            for line in lines:
                if line:  # Add any non-empty line as a potential question
                    processed_follow_ups.append(line)
        # Limit to 3 questions
        processed_follow_ups = processed_follow_ups[:3]
        # Ensure numbering for frontend consistency
        processed_follow_ups = [f"{i+1}. {q}" if not q.startswith(f"{i+1}.") else q for i, q in enumerate(processed_follow_ups)]
        # Fallback: If still empty, provide default follow-ups
        if not processed_follow_ups:
            processed_follow_ups = [
                "1. Can you provide more details on the budget allocation?",
                "2. How does this compare to the previous year's budget?",
                "3. What are the expected economic impacts?"
            ]
        print(f"Processed follow-ups: {processed_follow_ups}")
        return {"follow_ups": processed_follow_ups}
    except Exception as e:
        print(f"Error in suggest_follow_ups: {str(e)}")
        # Return default follow-ups on error
        default_follow_ups = [
            "1. Can you provide more details on the budget allocation?",
            "2. How does this compare to the previous year's budget?",
            "3. What are the expected economic impacts?"
        ]
        return {"follow_ups": default_follow_ups}

@app.route("/store_chat", methods=["POST"])
def store_chat():
    if "email" not in session:
        return {"error": "Unauthorized"}, 401
    email = session["email"]
    question = request.json["question"]
    answer = request.json["answer"]
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO history (email, question, answer, timestamp, starred) VALUES (?, ?, ?, ?, ?)",
              (email, question, answer, timestamp, 0))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.route("/feedback", methods=["POST"])
def feedback():
    if "email" not in session:
        return {"error": "Unauthorized"}, 401
    email = session["email"]
    question = request.json["question"]
    answer = request.json["answer"]
    rating = request.json["rating"]
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO feedback (email, question, answer, rating) VALUES (?, ?, ?, ?)",
              (email, question, answer, rating))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.route("/history")
def history():
    if "email" not in session:
        return redirect(url_for("login"))
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE email = ? ORDER BY timestamp DESC", (session["email"],))
    chats = c.fetchall()
    conn.close()
    return render_template("history.html", chats=chats)

@app.route("/star", methods=["POST"])
def star():
    if "email" not in session:
        return {"error": "Unauthorized"}, 401
    id = request.json["id"]
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE history SET starred = 1 - starred WHERE id = ? AND email = ?", (id, session["email"]))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.route("/test_groq")
def test_groq():
    try:
        groq_llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
        result = groq_llm.invoke("Test: What is 1+1?")
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(debug=True)