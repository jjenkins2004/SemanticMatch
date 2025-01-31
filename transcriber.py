import whisper

def getTranscript(fileName, language):

    #load the model
    model = whisper.load_model("small")

    #format the audio
    audio = whisper.load_audio(fileName)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model (more formatting basically)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    #set the correct language
    options = whisper.DecodingOptions(language=language)

    #transcribe audio file
    print("beginning transcription...")
    result = whisper.decode(model, mel, options)

    return result.text