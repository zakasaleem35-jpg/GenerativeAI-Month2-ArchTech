\---



\## Task 3: Implement RAG With Unsloth's Dynamic 4-bit Quantization



\*\*Folder:\*\* \[`task3-rag-pipeline/`](./task3-rag-pipeline)



A Retrieval-Augmented Generation (RAG) pipeline using Unsloth's dynamic 4-bit quantized Llama 3.2 (3B) model, combined with FAISS-based similarity search over a small domain-specific knowledge base, so answers are grounded in retrieved context rather than the model's raw training.



\### Tech Stack

\- Google Colab (T4 GPU)

\- Unsloth (dynamic 4-bit quantized `unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit`)

\- Sentence-Transformers (`all-MiniLM-L6-v2`)

\- FAISS



\### How to Run

1\. Open a Colab notebook with a T4 GPU runtime.

2\. Install dependencies: `pip install -r requirements.txt`

3\. Run `rag\_pipeline.py` cell by cell.



\### Results

\- Successfully indexed 8 documents and retrieved correct context for test queries

\- Model produced grounded, accurate answers based on retrieved passages



\---



\## Task 4: Build A Speech-to-Reasoning Pipeline With Whisper \& Quantized LLM



\*\*Folder:\*\* \[`task4-speech-reasoning/`](./task4-speech-reasoning)



An end-to-end pipeline: a spoken audio query is transcribed to text using OpenAI's Whisper, then passed to the same dynamic 4-bit quantized Llama 3.2 model for step-by-step reasoning.



\### Tech Stack

\- Google Colab (T4 GPU)

\- OpenAI Whisper (`base` model)

\- gTTS (for generating a sample spoken query)

\- Unsloth (dynamic 4-bit quantized `unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit`)



\### How to Run

1\. Open a Colab notebook with a T4 GPU runtime.

2\. Install dependencies: `pip install -r requirements.txt`

3\. Run `speech\_to\_reasoning.py` cell by cell.



\### Results

\- Whisper correctly transcribed the spoken query

\- Reasoning model produced a correct, step-by-step answer (15% of 240 = 36, which is more than 30)

