from app.rag.ingest import build_faiss_index

if __name__ == "__main__":
    print(build_faiss_index(lang="en"))
    print(build_faiss_index(lang="de"))
