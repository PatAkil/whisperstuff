build:
	docker build -t whisper-api .

run:
	docker run -p 8008:8008 whisper-api