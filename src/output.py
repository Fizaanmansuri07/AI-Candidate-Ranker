"""
Output formatters: CSV, JSON, and human-readable Markdown report
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from src.ranker import ScoredCandidate, JobDescription


# ─── CSV Output ──────────────────────────────────────────────────────────────────

def to_csv(ranked: list[ScoredCandidate], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "tier", "candidate_id", "name", "email",
            "total_score", "skill_score", "experience_score",
            "education_score", "culture_fit_score", "growth_potential_score",
            "logistics_score", "years_experience", "education",
            "top_skills", "strengths", "gaps", "recommendation",
        ])
        for s in ranked:
            c = s.candidate
            writer.writerow([
                s.rank,
                s.tier,
                c.id,
                c.name,
                c.email,
                s.total_score,
                s.skill_score,
                s.experience_score,
                s.education_score,
                s.culture_fit_score,
                s.growth_potential_score,
                s.logistics_score,
                c.years_experience,
                c.education,
                " | ".join(c.skills[:6]),
                " | ".join(s.strengths),
                " | ".join(s.gaps),
                s.recommendation,
            ])
    print(f"✓ CSV output → {path}")


# ─── JSON Output ─────────────────────────────────────────────────────────────────

def to_json(ranked: list[ScoredCandidate], job: JobDescription, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "job_title": job.title,
            "company": job.company,
            "total_candidates": len(ranked),
            "tier_distribution": _tier_dist(ranked),
        },
        "rankings": [],
    }
    for s in ranked:
        c = s.candidate
        output["rankings"].append({
            "rank": s.rank,
            "tier": s.tier,
            "candidate": {
                "id": c.id,
                "name": c.name,
                "email": c.email,
                "years_experience": c.years_experience,
                "education": c.education,
                "education_level": c.education_level,
                "skills": c.skills,
                "location": c.location,
                "remote_preference": c.remote_preference,
                "companies": c.companies,
                "previous_titles": c.previous_titles,
                "certifications": c.certifications,
                "open_source": c.open_source,
                "publications": c.publications,
                "github_url": c.github_url,
            },
            "scores": {
                "total": s.total_score,
                "skill": s.skill_score,
                "experience": s.experience_score,
                "education": s.education_score,
                "culture_fit": s.culture_fit_score,
                "growth_potential": s.growth_potential_score,
                "logistics": s.logistics_score,
            },
            "breakdown": s.score_breakdown,
            "strengths": s.strengths,
            "gaps": s.gaps,
            "recommendation": s.recommendation,
        })
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✓ JSON output → {path}")


# ─── Markdown Report ─────────────────────────────────────────────────────────────

def to_markdown_report(ranked: list[ScoredCandidate], job: JobDescription, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    tier_dist = _tier_dist(ranked)
    top10 = ranked[:10]

    lines = [
        f"# 🎯 Candidate Ranking Report",
        f"",
        f"**Role:** {job.title} @ {job.company}  ",
        f"**Generated:** {now}  ",
        f"**Total Candidates Evaluated:** {len(ranked)}",
        f"",
        f"---",
        f"",
        f"## Tier Distribution",
        f"",
        f"| Tier | Count | % | Label |",
        f"|------|-------|---|-------|",
        f"| 🥇 S | {tier_dist.get('S',0)} | {tier_dist.get('S',0)/len(ranked)*100:.1f}% | Strong Hire |",
        f"| 🥈 A | {tier_dist.get('A',0)} | {tier_dist.get('A',0)/len(ranked)*100:.1f}% | Hire |",
        f"| 🥉 B | {tier_dist.get('B',0)} | {tier_dist.get('B',0)/len(ranked)*100:.1f}% | Consider |",
        f"| ⚪ C | {tier_dist.get('C',0)} | {tier_dist.get('C',0)/len(ranked)*100:.1f}% | Borderline |",
        f"| ❌ D | {tier_dist.get('D',0)} | {tier_dist.get('D',0)/len(ranked)*100:.1f}% | Pass |",
        f"",
        f"---",
        f"",
        f"## Top 10 Candidates",
        f"",
        f"| Rank | Tier | Name | Score | Exp | Skill | Culture | Recommendation |",
        f"|------|------|------|-------|-----|-------|---------|----------------|",
    ]

    for s in top10:
        c = s.candidate
        lines.append(
            f"| #{s.rank} | {s.tier} | {c.name} | {s.total_score:.3f} | "
            f"{c.years_experience}y | {s.skill_score:.2f} | "
            f"{s.culture_fit_score:.2f} | {s.recommendation.split('—')[0].strip()} |"
        )

    lines += ["", "---", "", "## Detailed Candidate Profiles (Top 20)", ""]

    for s in ranked[:20]:
        c = s.candidate
        skill_match = s.score_breakdown["skill"]
        lines += [
            f"### #{s.rank} — {c.name} `[Tier {s.tier}]`",
            f"",
            f"**Score:** `{s.total_score:.4f}` | **Recommendation:** _{s.recommendation}_",
            f"",
            f"| Dimension | Score | Weight |",
            f"|-----------|-------|--------|",
            f"| 🔧 Skill Match | {s.skill_score:.3f} | 35% |",
            f"| 📅 Experience | {s.experience_score:.3f} | 25% |",
            f"| 🏫 Education | {s.education_score:.3f} | 10% |",
            f"| 🤝 Culture Fit | {s.culture_fit_score:.3f} | 15% |",
            f"| 📈 Growth Potential | {s.growth_potential_score:.3f} | 10% |",
            f"| 🗺️ Logistics | {s.logistics_score:.3f} | 5% |",
            f"",
            f"**Profile:**  ",
            f"- Experience: {c.years_experience} years | Education: {c.education}",
            f"- Location: {c.location} | Remote: {c.remote_preference}",
            f"- Companies: {', '.join(c.companies[:3])}",
            f"- Skills: {', '.join(c.skills[:8])}",
            f"",
        ]
        if s.strengths:
            lines.append("**✅ Strengths:**")
            for strength in s.strengths:
                lines.append(f"- {strength}")
            lines.append("")
        if s.gaps:
            lines.append("**⚠️ Gaps:**")
            for gap in s.gaps:
                lines.append(f"- {gap}")
            lines.append("")
        if skill_match.get("required_missing"):
            lines.append(f"**Missing Required Skills:** {', '.join(skill_match['required_missing'][:5])}")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines += [
        f"## Score Weights Reference",
        f"",
        f"| Dimension | Weight | Rationale |",
        f"|-----------|--------|-----------|",
        f"| Skill Match | 35% | Directly predicts ability to do the job |",
        f"| Experience | 25% | Depth and relevance of past work |",
        f"| Education | 10% | Foundation knowledge signal |",
        f"| Culture Fit | 15% | Company fit, impact history, ownership signals |",
        f"| Growth Potential | 10% | Trajectory, publications, open source |",
        f"| Logistics | 5% | Location, remote preference, salary alignment |",
        f"",
        f"*Generated by AI Candidate Ranking System — NeuralRank v1.0*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).encode("utf-8", errors="ignore").decode("utf-8"))
    print(f"✓ Markdown report → {path}")


def _tier_dist(ranked: list[ScoredCandidate]) -> dict:
    dist = {}
    for s in ranked:
        dist[s.tier] = dist.get(s.tier, 0) + 1
    return dist
