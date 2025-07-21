# Check if documents are in the vector store
from local_vector_store import LocalVectorStore
vs = LocalVectorStore()
stats = vs.get_stats()
print(f"Documents in store: {stats}")

# Test search directly
results = vs.search("vacation days", k=5, score_threshold=0.1)
print(f"Search results: {len(results)}")
for r in results:
    print(f"Score: {r['score']}, Text: {r['text'][:100]}")