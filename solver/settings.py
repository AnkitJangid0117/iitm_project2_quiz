import os

class Settings:
    STUDENT_SECRET = os.getenv("STUDENT_SECRET")
    STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")
    BROWSERLESS_KEY = os.getenv("BROWSERLESS_KEY")

settings = Settings()
