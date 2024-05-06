from sentence_transformers import SentenceTransformer
import numpy as np

def split_into_chunks(sentence, chunk_len=768):
    chunks = []
    for i in range(0, len(sentence), chunk_len):
        chunks.append(sentence[i:i+chunk_len])
    return chunks

def embed(doc, candidates):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_splits = split_into_chunks(doc)
    doc_embedding = np.mean(model.encode(doc_splits), axis=0).reshape(1, -1)
    candidate_embeddings = model.encode(candidates)
    return doc_embedding, candidate_embeddings