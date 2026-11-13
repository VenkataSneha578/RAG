from langchain_community.document_loaders import WebBaseLoader
from helper import ask_from_docs

loader=WebBaseLoader("https://docs.langchain.com/build-overview")
docs=loader.load()
question =" Total how many types of chains are there in langchain?"
answer = ask_from_docs(docs, question)
print(answer)
# print(docs[0].page_content[:2000])
# print(docs[0].metadata)