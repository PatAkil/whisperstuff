from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
import whisper
import torch
from pydub import AudioSegment
from pydub.utils import make_chunks

# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model:
model = whisper.load_model("base", device=DEVICE)

app = Flask(__name__)


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
        sound = AudioSegment(handle)

        chunk_length_ms = 1000
        chunks = make_chunks(sound, chunk_length_ms)

        transcriptions = []
        for i, chunk in enumerate(chunks):
            temp = NamedTemporaryFile()
            chunk.save(temp)
            result = model.transcribe(temp.name)
            transcriptions.append(result['text'])

        results.append({
            'filename': filename,
            'transcript': transcriptions,
        })

    return {'results': results}
