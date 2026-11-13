from langchain_community.document_loaders import TextLoader

from helper import ask_from_docs


loader = TextLoader(r"/Users/venkatasnehanetuluri/RAG/todaynews.txt",encoding="utf-8")
docs= loader.load()

# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

question = "What happened to hardik pandya? why he is not selected into indian squad?"

answer =ask_from_docs(docs, question)
print(answer)