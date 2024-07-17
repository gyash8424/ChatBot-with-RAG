from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_chroma import Chroma
from langchain.llms import CTransformers
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()
docsearch = Chroma(persist_directory="D:\coding\projects\ChatBot\MediBot", embedding_function=embeddings)


llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q5_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':128,
                          'temperature':0.8})

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=rag_chain.invoke({"question": input, 'input':input})
    print("Response : ", result["answer"])
    return str(result["answer"])

if __name__ == '__main__':
    app.run(debug= True)