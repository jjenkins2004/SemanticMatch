from sentence_transformers import SentenceTransformer, util

def find_similarity(transcription, reference):
    #load the model
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    #find embeddings
    transcriptionEmbedding = model.encode(transcription)
    referenceEmbedding = model.encode(reference)

    similarity = util.cos_sim(transcriptionEmbedding, referenceEmbedding)

    return similarity.item()