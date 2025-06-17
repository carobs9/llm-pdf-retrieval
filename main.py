import os
import json
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
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
from pathlib import Path
import config as cfg

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

input_path = cfg.INPUT_PATH
output_path = cfg.OUTPUT_PATH 

model_name = cfg.MODEL_NAME
parser_usage = cfg.PARSER_USAGE

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

# 2. Load embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text') 


# 3. Load Mistral model through API
if cfg.MODEL_NAME == 'mistral':
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    instructor_client = from_mistral(
        client=client,
        model="mistral-large-latest",
        mode=Mode.MISTRAL_TOOLS,
        max_tokens=1000,
    )

    results = []

    for filename in input_path.iterdir():
        if filename.suffix.lower() != ".pdf":
            continue

        print(f"Processing {filename}")
        file_path = filename

        try:
            # 4. Process PDFs and convert them to readable input
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            pages = loader.load_and_split()

            store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
            retriever = store.as_retriever()
            relevant_docs = retriever.get_relevant_documents("What is this paper about?")
            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            metadata = docs[0].metadata
            # 5. Get response
            # client = instructor.from_provider(f"mistral/mistral")
            resp = instructor_client.messages.create(
                    response_model=Structure,
                    messages=[{"role": "user", "content": context}],
                    temperature=0,
                )
            
            # 6. Append results
            results.append({
                "filename": str(filename),
                "output": resp.model_dump()
            })

        except Exception as e:
            results.append({
                "filename": str(filename),
                "output": f"ERROR: {str(e)}"
                })
            
output_file = output_path / "output.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("--- Extraction completed in %s seconds ---" % (time.time() - start_time))
print(f"Results saved to {output_path}")
sys.exit()