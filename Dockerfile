FROM python:3.10-slim

WORKDIR /whisper-api

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install git -y
RUN pip3 install -r requirements.txt
RUN pip3 install "git+https://github.com/openai/whisper.git" 
RUN apt-get install -y ffmpeg

COPY . .

EXPOSE 8008

CMD [ "python3", "-m" , "flask", "run", "--host=127.0.0.1"]