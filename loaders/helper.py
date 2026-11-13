import os

# Chroma may fork worker processes after tokenizers have initialized.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0.2,
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ask_from_docs(docs, question):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    db = Chroma.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the context below. If the question can't be answered based on the context, say "I don't know".

Context:
{context}

Question: {question}
"""
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer
