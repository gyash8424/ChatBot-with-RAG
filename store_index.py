from src.helper import load_pdf,text_split,download_hugging_face_embeddings
from langchain_chroma import Chroma


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

docsearch = Chroma.from_documents(documents=text_chunks,embedding=embeddings,persist_directory='D:\coding\projects\ChatBot\MediBot')