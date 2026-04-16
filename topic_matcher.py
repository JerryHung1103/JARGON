from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_topics(new_topic, topics_list, top_k=5):
    if len(topics_list) == 0:
        return []
    existing_topics = [item["Topic"] for item in topics_list]
    model = SentenceTransformer(
        'all-MiniLM-L6-v2', 
        cache_folder='./SentenceTransformer', 
        device='cpu'# Use CPU, adjust as needed
    )  
    
    topic_embeddings = model.encode(existing_topics)
    new_topic_embedding = model.encode([new_topic])
    

    similarities = cosine_similarity(new_topic_embedding, topic_embeddings)[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        print("topic:", existing_topics[idx], 'Sim: ', float(similarities[idx]))
        results.append({
            "topic": existing_topics[idx],
            "conversation": topics_list[idx]["Conversation"]
        })
    return results

