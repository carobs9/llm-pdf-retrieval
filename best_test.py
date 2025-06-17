import os
import json
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from sys import argv
import sys
import time
import pickle
import instructor
from pydantic import BaseModel
from mistralai import Mistral
from instructor import from_mistral, Mode
import pickle

start_time = time.time()

folder_path = r'C:\Users\rqg886\Desktop\LLM-pdf-extraction\pdfs'
parser_usage = True
MODEL_NAME = 'mistral'

# 1. Schema definition
class PDFInfo(BaseModel):
    title: str | None = Field(description="Title of the paper")
    religion: str | None = Field(description="Religion(s) referenced")
    country: str | None = Field(description="Country studied")
    key_results: str | None = Field(description="Key results or findings")
    methodology: str | None = Field(description="Study methodology")
    sample_size: str | None = Field(description="Sample size of the study")

class Structure(BaseModel):
    title: str
    abstract: str
    pages: int
    religion: str
    country: str
    results: str


embeddings = OllamaEmbeddings(model='nomic-embed-text') 

old_prompt = ChatPromptTemplate.from_messages([
      ("system", """
  You are an extraction assistant. Your only task is to extract the following information from academic paper text. Please, do not include any comments or annotations. Structure the results
  only as JSON format, where the keys are the following points, and the values are your answers:
  1. What is the title of the document?
  2. What is the document about?
  3. How many pages dies the document have?
  4. What religion is the document talking about?
  """),
      ("human", "{text}")
  ])

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a precise information extraction assistant.

Your only task is to extract the following information from the academic paper text and return it as a **valid JSON object**.

⚠️ Important:
- Do **not** include any comments, explanations, or markdown formatting.
- Do **not** write phrases like "Here is the JSON" or use triple backticks.
- Only output the JSON object itself.

Your JSON keys must be exactly:
- "title"
- "about"
- "num_pages"
- "religion"

Return this JSON using the information extracted from the provided text.
"""),
    ("human", "{text}")
])


# 3. Load PDF and create retriever
if MODEL_NAME == 'mistral':
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    instructor_client = from_mistral(
        client=client,
        model="mistral-large-latest",
        mode=Mode.MISTRAL_TOOLS,
        max_tokens=1000,
    )

    results = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        print(f"Processing {filename}")
        file_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        pages = loader.load_and_split()

        store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
        retriever = store.as_retriever()
        relevant_docs = retriever.get_relevant_documents("What is this paper about?")
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        metadata = docs[0].metadata
        client = instructor.from_provider(f"mistral/mistral")
        resp = instructor_client.messages.create(
                response_model=Structure,
                messages=[{"role": "user", "content": context}],
                temperature=0,
            )
        print(resp)

        print("--- %s seconds ---" % (time.time() - start_time))
        
elif MODEL_NAME == 'llama3':
                # 2. LLM + Embeddings
    llm = OllamaLLM(
                model=MODEL_NAME,
                temperature=0,
                # other params...
        )

    results = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        print(f"Processing {filename}")
        file_path = os.path.join(folder_path, filename)
        
        try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                pages = loader.load_and_split()
                store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
                retriever = store.as_retriever()
                query = "What is this paper about?"
                relevant_docs = retriever.get_relevant_documents(query)
                context = "\n\n".join(doc.page_content for doc in relevant_docs)

                if parser_usage is True: # user parser 
                    parser = PydanticOutputParser(pydantic_object=PDFInfo)
                    chain = prompt | llm | parser
                else:
                    chain = prompt | llm

                    output = chain.invoke({
                            "text": context,
                            "author": docs[0].metadata.get("author", "unknown"),
                            "year": docs[0].metadata.get("year", "unknown"),
                            "subject": docs[0].metadata.get("subject", "unknown"),
                            "title": docs[0].metadata.get("title", "unknown"),
                            "number of pages": len(docs)
                        })

                    results.append({
                        "filename": filename,
                        "output": output
                    })
                    
        except Exception as e:
            results.append({
                "filename": filename,
                "output": f"ERROR: {str(e)}"
                })

        with open(f"results_{MODEL_NAME}_parser" if parser_usage is True else f"results_{MODEL_NAME}", "wb") as fp:   #Pickling
            pickle.dump(results, fp)


        print("--- %s seconds ---" % (time.time() - start_time))



sys.exit()