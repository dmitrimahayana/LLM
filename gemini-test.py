import os
import google.generativeai as genai

# Read API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)

# Choose model
model = genai.GenerativeModel("gemini-2.5-flash")

# Simple test prompt
response = model.generate_content("Explain equilibrium in economics in 3 sentences.")

print("=== Gemini Response ===")
print(response.text)
