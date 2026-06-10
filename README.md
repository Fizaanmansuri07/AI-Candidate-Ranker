# 🎯 NeuralRank — AI-Powered Candidate Ranking System

> **Hackathon Track 1 Submission** — Intelligent, explainable candidate ranking that goes beyond keyword filters.

[![Tests](https://img.shields.io/badge/tests-23%2F23%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)]()
[![Speed](https://img.shields.io/badge/speed-13%2C000%2B%20candidates%2Fsec-orange)]()

---

## The Problem With Keyword Filters

Most ATS systems rank candidates by counting keyword matches. This breaks down immediately:

- A resume listing "Python" 8 times outranks a 10-year ML architect who used a different abbreviation
- Overqualified candidates get auto-rejected
- There's no explanation for any ranking decision

**NeuralRank** replaces keyword counting with a **6-dimensional scoring engine** that evaluates each candidate holistically and explains every decision.

---

## Quick Start

```bash
git clone https://github.com/your-username/ai-candidate-ranker
cd ai-candidate-ranker

# No install needed — zero external dependencies
python main.py
```

This generates 500 sample candidates, ranks them for a "Senior ML Engineer" role, and outputs:
- `output/ranked_candidates.csv` — Machine-readable rankings
- `output/ranked_candidates.json` — Full structured output with score breakdowns
- `output/ranking_report.md` — Human-readable explainable report

### Using Your Own Data

```bash
# Bring your own candidates and job description
python main.py --candidates data/candidates.json --job data/job.json

# Options
python main.py --n 1000          # Generate N sample candidates
python main.py --top 50          # Only include top 50 in output
python main.py --format json     # csv | json | md | all
python main.py --seed 123        # Reproducible results
```

---

## How It Works

### 6-Dimensional Scoring

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| 🔧 Skill Match | **35%** | Semantic skill coverage, weighted by importance |
| 📅 Experience | **25%** | Years + seniority calibration + title relevance |
| 🏫 Education | **10%** | Level meets requirement + field relevance |
| 🤝 Culture Fit | **15%** | Company prestige, achievement language, certifications |
| 📈 Growth Potential | **10%** | Publications, open source, career trajectory |
| 🗺️ Logistics | **5%** | Remote alignment, salary fit |

### Tier System

```
[S] ≥ 0.85  →  Strong Hire — Fast-track to final round
[A] ≥ 0.70  →  Hire        — Schedule technical interview
[B] ≥ 0.55  →  Consider    — Phone screen recommended
[C] ≥ 0.40  →  Borderline  — Review manually
[D] < 0.40  →  Pass        — Does not meet requirements
```

### Explainable Output

Every candidate ranking comes with a full breakdown:

```json
{
  "rank": 1,
  "tier": "A",
  "candidate": { "name": "Arjun Garcia", ... },
  "scores": {
    "total": 0.8376,
    "skill": 0.8120,
    "experience": 0.9500,
    "education": 0.8500
  },
  "strengths": [
    "Strong skill match (96% of required skills)",
    "8.5 years experience — meets requirement",
    "Active open-source contributor"
  ],
  "gaps": [
    "Missing key skills: large language models"
  ],
  "recommendation": "Hire — Schedule technical interview"
}
```

---

## Key Features

**Semantic Skill Normalization**  
"LLMs", "Large Language Models", "large language model" → all match the same canonical form. 18 synonym groups built in; trivially extensible.

**Weighted Skill Importance**  
A role requiring "Python" and "RLHF" cares more about the rare skill. Skill weights encode domain-specific importance.

**Nuanced Experience Scoring**  
Overqualified candidates get mild penalties, not hard rejections. Under-qualification is penalized proportionally to the gap, not with a binary pass/fail.

**Zero Dependencies**  
Runs on Python 3.10+ standard library only. No `pip install`, no rate limits, no API keys.

**13,000+ candidates/second**  
Ranks a 500-candidate pool in 40ms on a standard laptop CPU.

---

## Project Structure

```
ai-candidate-ranker/
├── main.py                   # CLI entry point
├── src/
│   ├── ranker.py             # Core scoring engine, data models
│   ├── data.py               # Dataset generation + loaders
│   └── output.py             # CSV, JSON, Markdown formatters
├── tests/
│   └── test_ranker.py        # 23 unit + integration tests
├── docs/
│   └── METHODOLOGY.md        # Detailed technical methodology
├── output/                   # Generated rankings (gitignored)
└── requirements.txt          # pytest only (optional)
```

---

## Data Schema

### Candidate JSON

```json
{
  "id": "CAND-0001",
  "name": "Alex Chen",
  "email": "alex.chen@email.com",
  "years_experience": 7.5,
  "skills": ["Python", "Machine Learning", "PyTorch", "MLOps"],
  "education": "MSc Computer Science",
  "education_level": "masters",
  "previous_titles": ["Senior ML Engineer", "ML Engineer"],
  "companies": ["Google", "Stripe"],
  "location": "San Francisco, CA",
  "remote_preference": "hybrid",
  "salary_expectation": 165000,
  "achievements": ["Reduced latency by 40%", "Led team of 5"],
  "certifications": ["AWS Certified ML Specialty"],
  "open_source": true,
  "publications": 2,
  "languages_spoken": ["English", "Mandarin"],
  "linkedin_url": "https://linkedin.com/in/alexchen",
  "github_url": "https://github.com/alexchen"
}
```

### Job Description JSON

```json
{
  "title": "Senior ML Engineer",
  "company": "NeuralPath Technologies",
  "required_skills": ["Python", "Machine Learning", "PyTorch", "MLOps"],
  "preferred_skills": ["Kubernetes", "Docker", "LangChain"],
  "min_years_experience": 5,
  "max_years_experience": 15,
  "education_requirement": "bachelors",
  "location": "San Francisco, CA",
  "remote_policy": "hybrid",
  "salary_range": [140000, 200000],
  "key_responsibilities": ["..."],
  "nice_to_haves": ["Publications", "Open source"],
  "seniority": "senior"
}
```

---

## Running Tests

```bash
# With pytest installed
python -m pytest tests/ -v

# Without pytest (stdlib runner)
python tests/test_ranker.py
```

**23/23 tests passing** covering:
- Skill synonym matching
- Perfect/partial/zero skill coverage
- Experience range boundary conditions
- Education requirement checking
- End-to-end ranking correctness
- Performance benchmark (1000 candidates < 10s)
- Reproducibility with seed

---

## Methodology

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for:
- Full scoring formulas
- Architectural decisions and trade-offs
- Why we chose rule-based over neural approaches
- Validation strategy
- Roadmap for production deployment

---

## License

MIT — free to use, fork, and extend.

---

*NeuralRank v1.0 — Built for Hackathon Track 1 by [Fizaan Mansuri]*
