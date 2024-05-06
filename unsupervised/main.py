import embeddings, keyphrases, similarity
import sys
import json

def readDoc(path):
    with open(path, 'r') as file:
        text = file.read()
    return text

def save(gen_dict):
    with open('extracted.json', 'w') as file:
        json.dump(gen_dict, file, indent = 4)

def generate(doc):
    minLength, maxLength = 1, 3
    k = 5
    nk = 20
    diversity = 0.2
    candidates = keyphrases.generateCandidates(doc, minLength, maxLength)
    doc_embeddings, candidate_embeddings = embeddings.embed(doc, candidates)

    topK = similarity.getTopKSimilar(k, doc_embeddings, candidate_embeddings, candidates)
    MSS = similarity.maxSumSimilar(doc_embeddings, candidate_embeddings, k, nk, candidates)
    MMR = similarity.maximalMarginalRelevance(doc_embeddings, candidate_embeddings, candidates, k, diversity)

    gen_dict = {'topK': topK, 'MSS': MSS, 'MMR': MMR}
    return gen_dict

if __name__ == '__main__':
    docPath = sys.argv[1]
    doc = readDoc(docPath)
    save(generate(doc))
    


