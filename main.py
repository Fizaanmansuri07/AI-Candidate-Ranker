#!/usr/bin/env python3
"""
NeuralRank — AI-Powered Candidate Ranking System
Usage:
  python main.py                          # Run on generated sample data
  python main.py --candidates data/candidates.json --job data/job.json
  python main.py --top 20 --format all
"""

import argparse
import json
import time
from pathlib import Path

from src.ranker import CandidateRanker
from src.data import (
    generate_dataset,
    sample_job_description,
    load_candidates_from_json,
    load_candidates_jsonl,
    load_job_from_json,
)
from src.output import to_csv, to_json, to_markdown_report


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║          NeuralRank — AI Candidate Ranker            ║
║        Intelligent Multi-Dimensional Scoring         ║
╚══════════════════════════════════════════════════════╝
""")


def print_summary(ranked, job):
    from src.output import _tier_dist
    dist = _tier_dist(ranked)
    top5 = ranked[:5]

    print(f"\n{'─'*60}")
    print(f"  Job: {job.title} @ {job.company}")
    print(f"  Evaluated: {len(ranked)} candidates")
    print(f"{'─'*60}")
    print(f"  Tier Distribution:")
    for tier, label in [("S","Strong Hire"),("A","Hire"),("B","Consider"),("C","Borderline"),("D","Pass")]:
        n = dist.get(tier, 0)
        bar = "█" * n + "░" * max(0, 30 - n)
        print(f"  [{tier}] {bar[:30]}  {n:4d}  {label}")
    print(f"\n  🏆 Top 5 Candidates:")
    print(f"  {'Rank':<5} {'Name':<22} {'Score':<8} {'Tier':<5} {'Recommendation'}")
    print(f"  {'─'*75}")
    for s in top5:
        rec_short = s.recommendation.split("—")[0].strip()
        print(f"  #{s.rank:<4} {s.candidate.name:<22} {s.total_score:.4f}   {s.tier:<5} {rec_short}")
    print(f"{'─'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="NeuralRank — AI Candidate Ranking System")
    parser.add_argument("--candidates", help="Path to candidates JSON file")
    parser.add_argument("--job", help="Path to job description JSON file")
    parser.add_argument("--n", type=int, default=500, help="Number of sample candidates to generate (default: 500)")
    parser.add_argument("--top", type=int, default=None, help="Only include top N in detailed output")
    parser.add_argument("--format", default="all", choices=["csv","json","md","all"], help="Output format")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print_banner()

        # ── Load data ──────────────────────────────────────────────────────────────
    if args.candidates:
        print(f"  Loading candidates from {args.candidates}...")

        if args.candidates.endswith(".jsonl"):
            candidates = load_candidates_jsonl(args.candidates)
        else:
            candidates = load_candidates_from_json(args.candidates)

    else:
        print(f"  Generating {args.n} sample candidates (seed={args.seed})...")
        candidates = generate_dataset(n=args.n, seed=args.seed)

    if args.job:
        print(f"  Loading job description from {args.job}...")
        job = load_job_from_json(args.job)
    else:
        print("  Using sample job description: Senior ML Engineer @ NeuralPath Technologies")
        job = sample_job_description()

    # Save input data for reference
    Path(args.out).mkdir(parents=True, exist_ok=True)
    _save_sample_data(candidates, job, args.out)

    # ── Run ranking ────────────────────────────────────────────────────────────
    print("\n  Running multi-dimensional scoring...")
    t0 = time.time()

    ranker = CandidateRanker(job)
    ranked = ranker.rank(candidates)

    elapsed = time.time() - t0
    print(f"  Ranked {len(ranked)} candidates in {elapsed:.2f}s ({len(ranked)/elapsed:.0f} candidates/sec) ✓")

    if args.top:
        ranked = ranked[:args.top]

    # ── Print summary ──────────────────────────────────────────────────────────
    print_summary(ranked, job)

    # ── Write outputs ──────────────────────────────────────────────────────────
    print("  Writing outputs...")
    fmt = args.format

    if fmt in ("csv", "all"):
        to_csv(ranked, f"{args.out}/ranked_candidates.csv")
    if fmt in ("json", "all"):
        to_json(ranked, job, f"{args.out}/ranked_candidates.json")
    if fmt in ("md", "all"):
        to_markdown_report(ranked, job, f"{args.out}/ranking_report.md")

    print(f"\n  ✅ All outputs written to ./{args.out}/")
    print(f"     ranked_candidates.csv  — Machine-readable rankings")
    print(f"     ranked_candidates.json — Full structured output with score breakdowns")
    print(f"     ranking_report.md      — Human-readable explainable report")


def _save_sample_data(candidates, job, out_dir):
    """Persist the generated dataset so it can be inspected or re-used."""
    cand_path = f"{out_dir}/sample_candidates.json"
    job_path  = f"{out_dir}/sample_job.json"

    cand_dicts = []
    for c in candidates:
        cand_dicts.append({
            "id": c.id, "name": c.name, "email": c.email,
            "years_experience": c.years_experience, "skills": c.skills,
            "education": c.education, "education_level": c.education_level,
            "previous_titles": c.previous_titles, "companies": c.companies,
            "location": c.location, "remote_preference": c.remote_preference,
            "salary_expectation": c.salary_expectation,
            "achievements": c.achievements, "certifications": c.certifications,
            "open_source": c.open_source, "publications": c.publications,
            "languages_spoken": c.languages_spoken,
            "linkedin_url": c.linkedin_url, "github_url": c.github_url,
        })
    with open(cand_path, "w") as f:
        json.dump(cand_dicts, f, indent=2)

    job_dict = {
        "title": job.title, "company": job.company,
        "required_skills": job.required_skills,
        "preferred_skills": job.preferred_skills,
        "min_years_experience": job.min_years_experience,
        "max_years_experience": job.max_years_experience,
        "education_requirement": job.education_requirement,
        "location": job.location, "remote_policy": job.remote_policy,
        "salary_range": list(job.salary_range) if job.salary_range else None,
        "key_responsibilities": job.key_responsibilities,
        "nice_to_haves": job.nice_to_haves,
        "seniority": job.seniority,
    }
    with open(job_path, "w") as f:
        json.dump(job_dict, f, indent=2)


if __name__ == "__main__":
    main()
