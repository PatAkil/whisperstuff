from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
import whisper
import torch
from pydub import AudioSegment
from pydub.utils import make_chunks
import openai
import gpt3

# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model:
model = whisper.load_model("base", device=DEVICE)

app = Flask(__name__)

openai.api_key = ""


@app.route("/")
def hello():
    return "Whisper Hello World!"


@app.route('/whisper', methods=['POST'])
def handler():
    if not request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        abort(400)

    results = []

    for filename, handle in request.files.items():
        print("handle", handle)
        print("filename", filename)

        sound = AudioSegment.from_mp3(handle)

        chunk_length_ms = 10000
        chunks = make_chunks(sound, chunk_length_ms)

        transcriptions = []
        for i, chunk in enumerate(chunks):
            temp = NamedTemporaryFile()
            chunk.export(temp)
            result = model.transcribe(temp.name, fp16=False)
            transcriptions.append(result['text'])

        summary = gpt3.askPrompt("Write a short summary of this podcast conversation: {}".format(transcriptions))
        results.append({
            'filename': filename,
            'transcript': transcriptions,
            'summary': summary,
        })

    return {'results': results}
