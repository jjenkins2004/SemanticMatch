import whisper
import numpy as np
import requests
import io
import subprocess

def getTranscript(fileLink, language):
    #get audio data from object link
    response = requests.get(fileLink)

    if (response.status_code != 200) :
        return ""
    
    #get the audio byte stream
    buffer = io.BytesIO(response.content)

    #load the model
    model = whisper.load_model("small")

    try :
        #load the audio to whisper format
        audio = load_audio(buffer)
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



def load_audio(buffer):
    #construct the ffmpeg command to convert io to OpenAI whisper format
    #note: '-i -' tells ffmpeg to read from stdin.
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", "-",      
        "-f", "s16le",    
        "-ac", "1",        
        "-acodec", "pcm_s16le",  
        "-ar", str(16000), 
        "-"                
    ]

    #ffmpeg takes stdin as input so we set stdin as the io buffer with input=buffer.getvalue()
    out = subprocess.run(cmd, input=buffer.getvalue(), capture_output=True, check=True).stdout

    #convert output to nparray according to whisper load_audio
    arr = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    #return nparray
    return arr
