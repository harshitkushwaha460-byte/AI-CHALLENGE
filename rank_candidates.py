import json
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from docx import Document

CANDIDATES_FILE = "candidates.jsonl"
JD_FILE = "job_description.docx"

RETRIEVAL_TERMS = [
    "retrieval","ranking","recommendation","search","vector","embedding",
    "embeddings","milvus","pinecone","weaviate","faiss","qdrant",
    "elasticsearch","opensearch","reranking","hybrid retrieval",
    "semantic search","ndcg","mrr","map","llm","lora","peft"
]
BAD_TITLES = [
    "hr manager","recruiter","marketing manager","sales",
    "accountant","content writer"
]

GOOD_TITLES = [
    "ai engineer",
    "machine learning engineer",
    "ml engineer",
    "applied scientist",
    "data scientist",
    "nlp engineer",
    "search engineer",
    "recommendation engineer",
    "software engineer",
    "backend engineer",
    "research engineer"
]

def role_penalty(c):
    title = c["profile"].get("current_title","").lower()
    return 0.3 if any(x in title for x in BAD_TITLES) else 1.0

def role_bonus(c):
    title = c["profile"].get("current_title", "").lower()

    for role in GOOD_TITLES:
        if role in title:
            return 1.15

    return 1.0

def read_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_candidates(path):

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # JSON format
    if content.startswith("["):
        return json.loads(content)

    # JSONL format
    data = []

    for line in content.splitlines():
        line = line.strip()

        if line:
            data.append(json.loads(line))

    return data

def build_text(c):
    p = c["profile"]
    chunks = [p.get("headline",""), p.get("summary","")]
    for job in c.get("career_history", []):
        chunks.append(job.get("title",""))
        chunks.append(job.get("description",""))
    for s in c.get("skills", []):
        chunks.append(s.get("name",""))
    return "\n".join(chunks)

def retrieval_score(c):
    text = build_text(c).lower()
    return sum(1 for t in RETRIEVAL_TERMS if t in text)

def career_score(c):

    text = build_text(c).lower()

    keywords = [
        "retrieval",
        "ranking",
        "recommendation",
        "search",
        "vector",
        "embedding",
        "embeddings",
        "llm",
        "milvus",
        "faiss",
        "pinecone",
        "semantic search",
        "reranking"
    ]

    return sum(
        1 for k in keywords
        if k in text
    )

def experience_score(c):
    y = c["profile"].get("years_of_experience", 0)
    if 5 <= y <= 9: return 1.0
    if 4 <= y < 5: return 0.8
    if 9 < y <= 11: return 0.7
    return 0.3

def signal_score(c):
    s = c.get("redrob_signals", {})
    score = 0.0
    score += s.get("profile_completeness_score",0)/100
    score += s.get("recruiter_response_rate",0)
    score += s.get("interview_completion_rate",0)
    score += max(0, s.get("offer_acceptance_rate",0))
    score += max(0, s.get("github_activity_score",-1))/100
    score += min(10,s.get("saved_by_recruiters_30d",0))/20
    if s.get("open_to_work_flag"): score += 0.3
    if s.get("willing_to_relocate"): score += 0.2
    return score

def reasoning(c):

    p = c["profile"]
    s = c.get("redrob_signals", {})

    title = p.get("current_title", "Unknown")
    years = p.get("years_of_experience", 0)

    skills = [
        x["name"]
        for x in c.get("skills", [])
    ]

    top_skills = []

    important = [
        "NLP",
        "Fine-tuning LLMs",
        "LoRA",
        "Milvus",
        "FAISS",
        "Pinecone",
        "Embeddings",
        "Ranking",
        "Recommendation Systems",
        "Vector Databases",
        "Python"
    ]

    for skill in skills:
        if skill in important:
            top_skills.append(skill)

    top_skills = top_skills[:3]

    response_rate = s.get(
        "recruiter_response_rate",
        0
    )

    open_to_work = s.get(
        "open_to_work_flag",
        False
    )

    reasons = []

    reasons.append(
        f"{years:.1f} years of experience as a {title}"
    )

    if top_skills:
        reasons.append(
            f"strong relevance in {', '.join(top_skills)}"
        )

    if response_rate > 0.5:
        reasons.append(
            f"high recruiter response rate ({response_rate:.0%})"
        )

    if open_to_work:
        reasons.append(
            "actively open to opportunities"
        )

    text = ". ".join(reasons) + "."

    if years > 10:
        text += (
            " Experience exceeds the preferred "
            "range but remains highly relevant."
        )

    return text

def rank_candidates(candidates_file, jd_file):

    jd_text = read_docx(jd_file).lower()

    candidates = load_candidates(
        candidates_file
    )
    print("Loaded candidates:", len(candidates))
    print("First candidate:", candidates[0]["candidate_id"])


    print("Fast filtering 100k candidates...")
    print("Total candidates:", len(candidates))
    print("First candidate:", candidates[0]["candidate_id"])
    print("Title:", candidates[0]["profile"]["current_title"])

    fast_scores = []

    for c in candidates:
        score = (
            3 * career_score(c)
            + 2 * retrieval_score(c)
            + signal_score(c)
            + experience_score(c)
        )

        fast_scores.append(score)

    print("Candidates:", len(candidates))
    print("Fast scores:", len(fast_scores))

    candidate_scores = list(zip(candidates, fast_scores))

    candidate_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    candidates = [
        c
        for c, _ in candidate_scores[:2000]
    ]

    print(
        f"Selected {len(candidates)} candidates "
        f"for semantic ranking."
    )

    texts = [build_text(c) for c in candidates]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    cand_emb = model.encode(texts, batch_size=256,
                            convert_to_numpy=True,
                            show_progress_bar=True)

    jd_emb = model.encode([jd_text], convert_to_numpy=True)[0]

    semantic = cosine_similarity(
        cand_emb, jd_emb.reshape(1, -1)
    ).flatten()

    semantic = MinMaxScaler().fit_transform(
        semantic.reshape(-1,1)
    ).flatten()

    retrieval = np.array([retrieval_score(c) for c in candidates])
    retrieval = MinMaxScaler().fit_transform(
        retrieval.reshape(-1,1)
    ).flatten()
    career = np.array([
    career_score(c)
    for c in candidates])
    career = MinMaxScaler().fit_transform(
    career.reshape(-1,1)).flatten()

    exp = np.array([experience_score(c) for c in candidates])
    sig = np.array([signal_score(c) for c in candidates])
    sig = MinMaxScaler().fit_transform(sig.reshape(-1,1)).flatten()

    scores = (
        0.40 * semantic +
        0.25 * career+
        0.15 * retrieval +
        0.10 * exp +
        0.10 * sig
    )
    scores = np.array([
    s * role_penalty(c) * role_bonus(c)
    for s, c in zip(scores, candidates)
])

    rows = []
    for c, s in zip(candidates, scores):
        rows.append({
            "candidate_id": c["candidate_id"],
            "score": float(s),
            "reasoning": reasoning(c)
        })
    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["score", "candidate_id"],
        ascending=[False, True]
    ).head(100)

    df["score"] = (
        df["score"] / df["score"].max()
    ).round(3)

    # Sort again after rounding to handle equal scores
    df = df.sort_values(
        by=["score", "candidate_id"],
        ascending=[False, True]
    )

    df["rank"] = range(1, len(df) + 1)

    submission = df[
        ["candidate_id", "rank", "score", "reasoning"]
    ]

    return submission


def main():

    submission = rank_candidates(
        "candidates.jsonl",
        "job_description.docx"
    )

    submission.to_csv(
        "minded_people.csv",
        index=False
    )

    print("Saved minded_people.csv")


if __name__ == "__main__":
    main()