from haystack_integrations.components.generators.ollama import OllamaGenerator

from haystack import Pipeline, Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore

template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?
"""

docstore = InMemoryDocumentStore()
docstore.write_documents([Document(content="World is a beautiful place"),
                          Document(content="Life is short"),
                          Document(content="Traveling is fun"),
                          Document(content="Make the world better place"),])

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
import gradio as gr

def answer_question(query):
    result = pipe.run({"prompt_builder": {"query": query},"retriever": {"query": query}})
    return result["llm"]["replies"][0]

iface = gr.Interface(fn=answer_question, inputs="text", outputs="text")
iface.launch()
