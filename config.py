import os

class Config:
    SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8080/tkm8")
    API_TOKEN = os.getenv("JWT_SECRET_B64", "6Lt2QQHWNLvi8ZhgGOeUAYkRXOcJDzaiOVn1xry1kFk=")