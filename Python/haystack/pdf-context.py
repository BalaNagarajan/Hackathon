from haystack_integrations.components.generators.ollama import OllamaGenerator

from haystack import Pipeline, Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
import gradio as gr
from PyPDF2 import PdfReader


template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?
"""

# Path to your local PDF file
pdf_path = "/Users/bala/DEV/Python/haystack/pdfs/cricket-in-india.pdf"

# Read PDF and create Haystack Document objects without using convert_files_to_documents

pdf_documents = []
with open(pdf_path, "rb") as f:
    reader = PdfReader(f)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_documents.append(Document(content=text))
# pdf_documents is already created above by reading the PDF and extracting text from each page.
# No need to use convert_files_to_documents.
# pdf_documents = convert_files_to_documents(dir_path=None, files=[pdf_path])

# Store documents in memory
docstore = InMemoryDocumentStore()
docstore.write_documents(pdf_documents)

generator = OllamaGenerator(model="mistral",
                            url = "http://localhost:11434",
                            generation_kwargs={
                              "num_predict": 100,
                              "temperature": 0.9,
                              })

pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=docstore))
pipe.add_component("prompt_builder", PromptBuilder(template=template, required_variables=["query", "documents"]))
pipe.add_component("llm", generator)
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

query = "Is Travelling bad"

result = pipe.run({"prompt_builder": {"query": query},"retriever": {"query": query}})



# Remove meta attributes and get clean result
clean_result = {
    "answer": result["llm"]["replies"][0]
}

import json
result_json = json.dumps(clean_result, indent=2)


# Or just print the answer directly
print("Answer:", result["llm"]["replies"][0])
#Expose this as a web service endpoint using Gradio
# Install Gradio if not already installed: pip install gradio
# Run the following code to create a simple web interface
# Make sure to run this in a virtual environment where you have installed the required packages


def answer_question(query):
    result = pipe.run({"prompt_builder": {"query": query},"retriever": {"query": query}})
    return result["llm"]["replies"][0]

iface = gr.Interface(fn=answer_question, inputs="text", outputs="text")
iface.launch()
