from sentence_transformers import SentenceTransformer

model_name = "BAAI/bge-base-en-v1.5"
local_path = "./model_weights/bge-base-en-v1.5"

print(f"Attempting to download {model_name} to {local_path}...")
try:
    model = SentenceTransformer(model_name)
    model.save(local_path)
    print(f"Model successfully downloaded and saved to {local_path}")
except Exception as e:
    print(f"Error downloading or saving model: {e}")