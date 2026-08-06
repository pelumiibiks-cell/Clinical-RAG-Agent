import sys
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
MALARIA_CONTEXT = BASE_DIR / "Malaria_db"
MALARIA_FAISS_INDEX = MALARIA_CONTEXT / "malaria_faiss.index"
MALARIA_METAD = MALARIA_CONTEXT / "malaria_metadata.pkl"
SIMILARITY_THRESHOLD = 0.4

model = SentenceTransformer(MODEL)

index = faiss.read_index(str(MALARIA_FAISS_INDEX))

with MALARIA_METAD.open("rb") as file:
    metadata = pickle.load(file)


def search(query : str , top_results : int = 5):
  embeddings =  model.encode([query], convert_to_numpy=True, normalize_embeddings=True) # Query comes in list

  distances, indicies = index.search(embeddings, top_results)

  print("---Top Results---")
  results = []
  found = False

  for rank, (score,idx) in enumerate(zip(distances[0], indicies[0]), start = 1): # Count the zip,close brackets
     if idx < 0 :
        continue
     if score > SIMILARITY_THRESHOLD :
        found = True
        entry = metadata[int(idx)]
        results.append(entry)

        print(f"{rank}\nScore / 1 : {score:.2f}")
        print(f"Source : {entry.get('Source')}")
        print(f"Page : {entry.get('Page Number')}")
        print("-" * 30)
        print(f"Content : {entry.get('Content')}")

  if not found:
     print("No relevant Results...")
  return results


def main() :
   while True :
      query = input("Speak wise one, What do you seek: ")
      if query.lower() == 'exit':
         print("Knowledge onto you wise one")
         sys.exit()
      if not query:
         ("But you must definitely aquire knowledge")
         continue
      search(query)


if __name__ == "__main__":
   main()
