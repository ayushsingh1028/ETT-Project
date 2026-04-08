import os
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

print("Searching for a working model...")
working_model = None

# List of models to try (preferring Flash and Pro versions)
models_to_try = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

for model_name in models_to_try:
    print(f"Testing {model_name}...", end=" ", flush=True)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("hi", generation_config={"max_output_tokens": 10})
        print("✅ SUCCESS")
        working_model = model_name
        break
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            print("❌ QUOTA EXCEEDED (429)")
        elif "404" in error_msg:
            print("❌ NOT FOUND (404)")
        else:
            print(f"❌ ERROR: {error_msg[:50]}...")
    time.sleep(1)  # Avoid hammering

if working_model:
    print(f"\nFOUND WORKING MODEL: {working_model}")
    with open("working_model.txt", "w") as f:
        f.write(working_model)
else:
    print("\nNO WORKING MODELS FOUND WITH CURRENT API KEY.")
