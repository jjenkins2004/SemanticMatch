# Transcription and Evaluation with OpenAI Whisper and Sentence Transformer

Can transcribe English (en) or Mandarin (zh) audio files and compare them to a inputted prompt to evaluate correctness using cosine-similarity.

Takes an object URl as input, and is compatible with all OpenAI whisper audio formats. This service works efficiently by passing the object URL directly into memory instead of writing to storage

Could be useful for automatic transcribing and grading of audio based responses.

Uses:
OpenAI whisper
Sentence Transformer paraphrase-multilingual-MiniLM-L12-v2
