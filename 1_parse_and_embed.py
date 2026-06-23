import json
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge"
CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates.jsonl")
JD_TEXT = """
Deep technical depth in modern ML systems - embeddings, retrieval, ranking, LLMs, fine-tuning.
Scrappy product-engineering attitude - willing to ship a working ranker in a week.
We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher.
The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems.
Weeks 1-3: Audit what we currently have (BM25 + rule-based).
Weeks 4-8: Ship a v2 ranking system (embeddings, hybrid retrieval, LLM re-ranking).
Weeks 9-12: Set up evaluation infrastructure (NDCG, MRR, MAP, offline-to-online correlation, A/B testing).
5-9 years of experience preferred.
Disqualifiers: Pure research environments without production deployment. LangChain only under 12 months. No production code in last 18 months.
Required: Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5).
Required: Production experience with vector databases (Pinecone, Weaviate, Qdrant, Milvus, FAISS).
Required: Strong Python.
Required: Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP).
"""

# Hard filter keywords (must have at least one to be considered)
AI_ML_KEYWORDS = ["ml", "machine learning", "ai", "artificial intelligence", "nlp", "llm", "embedding", "retrieval", "ranking", "data", "python", "software engineer", "backend", "search"]

def create_candidate_text(c):
    # Combine profile info into a dense text representation
    profile = c.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    exp = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    
    skills = [s.get("name", "") for s in c.get("skills", []) if s.get("proficiency") in ["advanced", "expert", "intermediate"]]
    skills_str = ", ".join(skills[:15]) # top 15 skills
    
    career_texts = []
    for job in c.get("career_history", [])[:3]: # top 3 jobs
        job_title = job.get("title", "")
        desc = job.get("description", "")
        career_texts.append(f"{job_title}: {desc}")
    career_str = " | ".join(career_texts)
    
    text = f"Title: {title}. Experience: {exp} years. Headline: {headline}. Summary: {summary}. Skills: {skills_str}. Recent roles: {career_str}."
    return text

def calculate_heuristic_score(c):
    # Base heuristic on redrob_signals
    signals = c.get("redrob_signals", {})
    score = 0.0
    
    # Response rate is a strong indicator of availability
    rr = signals.get("recruiter_response_rate", 0)
    score += rr * 2.0
    
    # Open to work
    if signals.get("open_to_work_flag", False):
        score += 1.0
        
    # Interview completion rate
    icr = signals.get("interview_completion_rate", 0)
    score += icr * 1.5
    
    # GitHub activity
    gh = signals.get("github_activity_score", -1)
    if gh > 0:
        score += (gh / 100.0) * 1.0
        
    # Profile completeness
    pc = signals.get("profile_completeness_score", 0)
    score += (pc / 100.0) * 0.5
    
    return score

def passes_filter(c):
    # Filter 1: Years of experience (we want senior, let's say at least 2.5 years, ideally 5-9)
    exp = c.get("profile", {}).get("years_of_experience", 0)
    if exp < 2.5:
        return False
        
    # Filter 2: Keywords
    text_to_search = (c.get("profile", {}).get("headline", "") + " " + c.get("profile", {}).get("summary", "") + " " + c.get("profile", {}).get("current_title", "")).lower()
    
    # Check skills
    skills = [s.get("name", "").lower() for s in c.get("skills", [])]
    text_to_search += " " + " ".join(skills)
    
    has_keyword = False
    for kw in AI_ML_KEYWORDS:
        if kw in text_to_search:
            has_keyword = True
            break
            
    if not has_keyword:
        return False
        
    return True

def main():
    print("Loading embedding model...")
    # Using a fast and effective open-source embedding model
    model_name = "BAAI/bge-base-en-v1.5" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(model_name, device=device)
    
    jd_embedding = model.encode(JD_TEXT, normalize_embeddings=True)
    
    filtered_candidates = []
    texts_to_embed = []
    
    print("Processing candidates JSONL...")
    with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        # Process in chunks
        for line in tqdm(f, desc="Filtering"):
            c = json.loads(line)
            if passes_filter(c):
                text = create_candidate_text(c)
                filtered_candidates.append({
                    "candidate_id": c["candidate_id"],
                    "heuristic_score": calculate_heuristic_score(c),
                    "text": text,
                    "profile": c.get("profile", {}),
                    "signals": c.get("redrob_signals", {})
                })
                texts_to_embed.append(text)
                
    print(f"Filtered down to {len(filtered_candidates)} candidates.")
    
    print("Generating embeddings...")
    embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    
    print("Calculating cosine similarities with JD...")
    # embeddings shape: (N, D), jd_embedding shape: (D,)
    similarities = np.dot(embeddings, jd_embedding)
    
    # Combine scores
    print("Ranking candidates...")
    results = []
    for i, c in enumerate(filtered_candidates):
        sim_score = similarities[i]
        # Normalize heuristic score roughly to 0-1 range to combine
        # Max heuristic score is around 6.0
        h_score_norm = min(c["heuristic_score"] / 6.0, 1.0)
        
        # Weighted combination: 70% semantic match, 30% heuristic signal match
        final_score = (0.7 * sim_score) + (0.3 * h_score_norm)
        
        results.append({
            "candidate_id": c["candidate_id"],
            "sim_score": float(sim_score),
            "h_score": float(c["heuristic_score"]),
            "final_score": float(final_score),
            "text": c["text"],
            "profile": c["profile"],
            "signals": c["signals"]
        })
        
    # Sort by final score
    results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Keep top 1000 for cross-encoder/reasoning stage
    top_1000 = results[:1000]
    
    print("Saving top 1000 candidates...")
    with open("top_1000_candidates.json", "w", encoding="utf-8") as f:
        json.dump(top_1000, f, indent=2)
        
    print("Stage 1 complete.")

if __name__ == "__main__":
    main()
