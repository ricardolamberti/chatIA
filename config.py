import os

class Config:
    SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8080/tkm8")
    API_TOKEN = os.getenv("API_TOKEN", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjbGllbnQtc2VydmljZSIsImNvbXBhbnkiOiJTVUIiLCJpYXQiOjE3NTY5MjQ3MzAsImV4cCI6MTc1NjkyODMzMH0.vi0YOLLVe80gZomdo6HfCL_jVs9Gg6hBD_MSn58KMco")