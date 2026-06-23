# Intelligent Candidate Discovery Pipeline

This repository contains the source code for a state-of-the-art Multi-Stage Retrieval-Augmented Generation (RAG) and Cross-Encoder pipeline designed for the Hack2Skill Data & AI Challenge.

## Architecture

Our architecture directly addresses the need for candidates with "production experience" over pure research by leveraging a highly specific semantic search pipeline rather than simple keyword matching or generic generative LLMs.

1. **Pre-filtering & Heuristics:** We stream the massive `candidates.jsonl` file, extracting structured data and computing a heuristic score based on explicitly provided Redrob signals (e.g., `recruiter_response_rate`, `github_activity`).
2. **Dense Embeddings (FAISS):** Candidate profiles are converted into comprehensive text summaries and embedded using `BAAI/bge-base-en-v1.5`. We perform an L2 inner product search via FAISS to retrieve the Top 1000 semantic matches against the Job Description.
3. **Cross-Encoder Reranking:** The Top 1000 candidates are strictly evaluated against the JD using `cross-encoder/ms-marco-MiniLM-L-6-v2`. This model acts as our "Expert Recruiter", penalizing generic research resumes and highly rewarding candidates with hands-on production engineering skills.
4. **Final Scoring:** The final candidate score is a weighted ensemble of the Cross-Encoder semantic fit, FAISS retrieval score, and Redrob behavioral signals.

## Running the Code

### Prerequisites
- Python 3.8+
- GPU (Optional but highly recommended for fast embeddings)

### Installation
```bash
pip install -r requirements.txt
```

### Execution
1. Place `candidates.jsonl` and the job description in the `dataset/` directory.
2. **Parse & Embed:**
   ```bash
   python 1_parse_and_embed.py
   ```
3. **Retrieve & Rank:**
   ```bash
   python 2_retrieve_and_rank.py
   ```
4. **Fix/Format Output:**
   ```bash
   python fix_csv.py
   ```
5. **Validate:**
   ```bash
   python dataset/validate_submission.py Type_2.csv
   ```
