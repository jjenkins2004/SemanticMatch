import whisper
import requests
import io

def getTranscript(fileLink, language):

    #get audio data from object link
    response = requests.get(fileLink)

    if (response.status_code != 200) :
        return ""
    
    #get the audio byte stream
    buffer = io.BytesIO(response.content)
    buffer.name = "file.webm"

    #load the model
    model = whisper.load_model("small")

    try :
        #load and format the audio
        audio = whisper.load_audio(buffer)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model (more formatting basically)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    except Exception as e:
        print(f"could not load audio data from {fileLink}: {e}")
        return ""

    #set the correct language
    options = whisper.DecodingOptions(language=language)

    #transcribe audio file
    print("beginning transcription...")
    result = whisper.decode(model, mel, options)

    return result.text