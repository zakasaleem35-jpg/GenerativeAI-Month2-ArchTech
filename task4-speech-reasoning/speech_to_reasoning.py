%%capture
!pip install unsloth
!pip install --upgrade --no-cache-dir "git+https://github.com/unslothai/unsloth.git"
!pip install -U openai-whisper
!pip install gTTS

import whisper

whisper_model = whisper.load_model("base")

from gtts import gTTS

text_to_speak = "What is 15 percent of 240, and is that more than 30?"

tts = gTTS(text=text_to_speak, lang='en')
tts.save("/content/sample_audio.mp3")

print("Audio file created at /content/sample_audio.mp3")

audio_path = "/content/sample_audio.mp3"

result = whisper_model.transcribe(audio_path)
transcribed_text = result["text"]

print("Transcribed Text:", transcribed_text)

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048

reasoning_model, reasoning_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(reasoning_model)

prompt = f"""Answer the following question with clear step-by-step reasoning.

Question: {transcribed_text}

Answer:"""

inputs = reasoning_tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = reasoning_model.generate(**inputs, max_new_tokens=250, use_cache=True)
response = reasoning_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

final_answer = response.split("Answer:")[-1].strip()

print("Transcribed Question:", transcribed_text)
print("\nModel's Reasoning/Answer:", final_answer)