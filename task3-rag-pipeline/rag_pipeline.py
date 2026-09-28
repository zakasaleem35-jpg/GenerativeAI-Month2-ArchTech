from unsloth import FastLanguageModel
import torch

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

documents = [
    "Generative AI refers to models that can create new content such as text, images, audio, and code by learning patterns from existing data.",
    "Large Language Models (LLMs) are trained on massive text datasets and use transformer architectures to predict and generate human-like text.",
    "Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model, allowing the model to fetch relevant information before generating a response, improving factual accuracy.",
    "Quantization reduces the precision of a model's weights (e.g., from 16-bit to 4-bit) to save memory and speed up inference, with minimal loss in output quality.",
    "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that trains small adapter layers instead of updating all of a model's parameters.",
    "Unsloth is a library that makes fine-tuning and running large language models significantly faster and more memory-efficient, especially on consumer GPUs.",
    "Dynamic 4-bit quantization selectively preserves higher precision for critical model parameters while aggressively compressing less important ones, improving accuracy compared to uniform quantization.",
    "Vector embeddings represent text as numerical vectors, allowing semantic similarity between pieces of text to be measured using distance metrics like cosine similarity.",
]

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedder.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

print(f"Indexed {index.ntotal} documents.")

def retrieve(query, top_k=2):
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

def rag_query(user_query, top_k=2):
    retrieved_docs = retrieve(user_query, top_k=top_k)
    context = "\n".join(retrieved_docs)

    prompt = f"""Answer the question based only on the following context. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {user_query}

Answer:"""

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    answer = response.split("Answer:")[-1].strip()
    return answer, retrieved_docs

question = "What is quantization and why is it useful?"
answer, sources = rag_query(question)
print("Question:", question)
print("Retrieved Context:", sources)
print("Answer:", answer)

question2 = "What does RAG stand for and how does it work?"
answer2, sources2 = rag_query(question2)
print("Question:", question2)
print("Retrieved Context:", sources2)
print("Answer:", answer2)