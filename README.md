# AI Candidate Ranking System

## Overview

This project ranks candidates for an AI/ML role using a hybrid ranking approach combining:

* Semantic similarity between candidate profiles and job description
* Retrieval and search relevance signals
* Career relevance score
* Experience score
* Recruiter and platform activity signals

The system generates a ranked CSV submission containing:

* candidate_id
* rank
* score
* reasoning

---

## Repository Structure

```text
app.py
rank_candidates.py
requirements.txt
sample_candidates.json
job_description.docx
submission_metadata.yaml
README.md
```


## Run Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser.

Upload:

* sample_candidates.json
* job_description.docx

Click **Run Ranking**.

The application will generate and allow download of:

```text
submission.csv
```

---

## Reproduce Submission CSV

Run:

```bash
python rank_candidates.py
```

This generates:

```text
minded_people.csv
```

using:

```text
candidates.jsonl
job_description.docx
```

---

## Sandbox

Streamlit Cloud deployment:
https://ai-challenge-no3kdn3jpncxvhv9hzkquv.streamlit.app/
<br>(Note: If You Find this app in sleeping mode,Please click on the "get this app back up" And wait some second while app is being ready)

---

## Runtime

Tested on:

* CPU-only machine
* 16 GB RAM

Ranking completes within challenge constraints.

---

## Ranking Pipeline

The ranking process is designed to efficiently handle very large candidate pools (up to 100,000 candidates) while remaining within the challenge compute constraints.

### Stage 1: Fast Candidate Filtering

A lightweight scoring function is applied to all candidates using:

* Career relevance score
* Retrieval keyword score
* Experience score
* Recruiter activity signals

Fast filtering score:

```text
3 × Career Score
+ 2 × Retrieval Score
+ Signal Score
+ Experience Score
```

The top **2,000 candidates** are selected from the full candidate pool for further processing.

This reduces computational cost significantly while retaining the most relevant candidates.

---

### Stage 2: Semantic Ranking

For the selected top 2,000 candidates:

1. Candidate profiles are converted into text representations.
2. A Sentence Transformer model (`all-MiniLM-L6-v2`) generates embeddings.
3. The job description is embedded using the same model.
4. Cosine similarity is computed between candidate embeddings and the job description embedding.

This produces a semantic relevance score measuring how closely each candidate matches the target role.

---

### Stage 3: Final Score Computation

The final ranking score combines multiple signals:

```text
Final Score =
0.40 × Semantic Similarity
+ 0.25 × Career Relevance
+ 0.15 × Retrieval Relevance
+ 0.10 × Experience Score
+ 0.10 × Recruiter Signals
```

Role-based adjustments are then applied:

* Bonus for AI/ML-related roles
* Penalty for non-relevant roles (HR, Marketing, Sales, etc.)

---

### Stage 4: Candidate Ranking

Candidates are sorted by:

1. Final score (descending)
2. Candidate ID (ascending for tie-breaking)

The top 100 candidates are exported as:

```text
submission.csv
```

This two-stage retrieval + semantic ranking architecture enables efficient processing of large candidate pools while maintaining ranking quality.

Final score is computed using:

```text
0.40 × Semantic Similarity
+ 0.25 × Career Relevance
+ 0.15 × Retrieval Relevance
+ 0.10 × Experience Score
+ 0.10 × Recruiter Signals
```

Additional role bonuses and penalties are applied based on candidate job titles.

The top candidates are ranked and exported as a submission CSV.
