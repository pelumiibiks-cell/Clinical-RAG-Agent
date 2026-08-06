from dotenv import load_dotenv
from google import genai
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.mal")
KEY = os.getenv("THE_KEY")
if not KEY :
    raise ValueError ("API-KEY not Found - Check .env file")

client = genai.Client(api_key=KEY)
