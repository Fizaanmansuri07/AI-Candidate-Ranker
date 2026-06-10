"""
AI-Powered Candidate Ranking System
Core ranking engine using multi-dimensional semantic scoring
"""

import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


# ─── Data Models ────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    id: str
    name: str
    email: str
    years_experience: float
    skills: list[str]
    education: str
    education_level: str
    previous_titles: list[str]
    companies: list[str]
    location: str
    remote_preference: str
    salary_expectation: Optional[int]
    achievements: list[str]
    certifications: list[str]
    open_source: bool
    publications: int
    languages_spoken: list[str]
    linkedin_url: Optional[str]
    github_url: Optional[str]

    # Redrob challenge fields
    headline: str = ""
    summary: str = ""
    current_title: str = ""
    current_company: str = ""
    current_industry: str = ""

    recruiter_response_rate: float = 0.0
    interview_completion_rate: float = 0.0
    github_activity_score: float = 0.0
    saved_by_recruiters_30d: int = 0

    open_to_work: bool = False
    willing_to_relocate: bool = False
    notice_period_days: int = 90

    raw: dict = field(default_factory=dict)


@dataclass
class JobDescription:
    title: str
    company: str
    required_skills: list[str]
    preferred_skills: list[str]
    min_years_experience: float
    max_years_experience: Optional[float]
    education_requirement: str    # "bachelors", "masters", "phd", "any"
    location: str
    remote_policy: str            # "remote", "hybrid", "onsite", "flexible"
    salary_range: Optional[tuple[int, int]]
    key_responsibilities: list[str]
    nice_to_haves: list[str]
    seniority: str                # "junior", "mid", "senior", "staff", "principal"


@dataclass
class ScoredCandidate:
    candidate: Candidate
    total_score: float
    rank: int
    tier: str                     # "S", "A", "B", "C", "D"
    skill_score: float
    experience_score: float
    education_score: float
    culture_fit_score: float
    growth_potential_score: float
    logistics_score: float
    score_breakdown: dict
    strengths: list[str]
    gaps: list[str]
    recommendation: str


# ─── Skill Taxonomy ──────────────────────────────────────────────────────────────

SKILL_SYNONYMS = {
    "python": ["py", "python3", "python2"],
    "javascript": ["js", "es6", "es2015", "ecmascript"],
    "typescript": ["ts"],
    "react": ["reactjs", "react.js"],
    "node": ["nodejs", "node.js"],
    "machine learning": ["ml", "machine-learning"],
    "deep learning": ["dl", "deep-learning"],
    "natural language processing": ["nlp", "text mining"],
    "kubernetes": ["k8s"],
    "postgresql": ["postgres", "psql"],
    "mongodb": ["mongo"],
    "elasticsearch": ["elastic", "es"],
    "amazon web services": ["aws", "amazon cloud"],
    "google cloud platform": ["gcp", "google cloud"],
    "microsoft azure": ["azure"],
    "continuous integration": ["ci", "ci/cd"],
    "continuous deployment": ["cd", "ci/cd"],
    "docker": ["containerization", "containers"],
    "large language models": ["llm", "llms", "large language model"],
    "retrieval augmented generation": ["rag"],
    "tensorflow": ["tf"],
}

SKILL_WEIGHTS = {
    # Core technical (high weight)
    "python": 1.2, "machine learning": 1.3, "deep learning": 1.3,
    "large language models": 1.4, "retrieval augmented generation": 1.4,
    "natural language processing": 1.3, "mlops": 1.3,
    # Secondary technical (medium weight)
    "tensorflow": 1.1, "pytorch": 1.1, "kubernetes": 1.1,
    "docker": 1.0, "aws": 1.0, "python": 1.0,
    # General (standard weight)
    "sql": 0.9, "git": 0.8, "linux": 0.8,
}

EDUCATION_SCORES = {
    "phd": 1.0,
    "masters": 0.85,
    "bachelors": 0.70,
    "associate": 0.50,
    "bootcamp": 0.45,
    "self-taught": 0.40,
    "other": 0.35,
}

EDUCATION_REQUIREMENTS = {
    "phd": ["phd"],
    "masters": ["phd", "masters"],
    "bachelors": ["phd", "masters", "bachelors"],
    "any": ["phd", "masters", "bachelors", "associate", "bootcamp", "self-taught", "other"],
}

SENIORITY_EXPERIENCE = {
    "junior": (0, 3),
    "mid": (2, 6),
    "senior": (5, 12),
    "staff": (8, 20),
    "principal": (12, 30),
}


# ─── Scoring Engine ──────────────────────────────────────────────────────────────

class CandidateRanker:
    """
    Multi-dimensional candidate ranking engine.
    
    Scoring dimensions:
      1. Skill Match (35%)      — semantic + weighted skill overlap
      2. Experience (25%)       — years + seniority calibration
      3. Education (10%)        — level + field relevance
      4. Culture Fit (15%)      — signals from titles, companies, achievements
      5. Growth Potential (10%) — trajectory, open source, publications
      6. Logistics (5%)         — location, salary, remote preference
    """

    WEIGHTS = {
    "skill": 0.40,
    "experience": 0.30,
    "education": 0.05,
    "culture_fit": 0.10,
    "growth_potential": 0.05,
    "logistics": 0.10,
}

    def __init__(self, job: JobDescription):
        self.job = job
        self._build_skill_index()

    def _normalize_skill(self, skill: str) -> str:
        s = skill.lower().strip()
        for canonical, variants in SKILL_SYNONYMS.items():
            if s in variants or s == canonical:
                return canonical
        return s

    def _build_skill_index(self):
        self.required_norm = {self._normalize_skill(s) for s in self.job.required_skills}
        self.preferred_norm = {self._normalize_skill(s) for s in self.job.preferred_skills}
        self.nice_norm = {self._normalize_skill(s) for s in self.job.nice_to_haves}

    def _candidate_skills_norm(self, candidate):
        skills = {
            self._normalize_skill(s)
            for s in candidate.skills
        }

        text = " ".join([
            candidate.headline or "",
            candidate.summary or "",
            candidate.current_title or "",
            " ".join(candidate.previous_titles or [])
        ]).lower()

        if "machine learning" in text:
            skills.add("machine learning")

        if "deep learning" in text:
            skills.add("deep learning")

        if "llm" in text or "large language model" in text:
            skills.add("large language models")

        if "rag" in text:
            skills.add("retrieval augmented generation")

        if "pytorch" in text:
            skills.add("pytorch")

        if "aws" in text:
            skills.add("amazon web services")

        return skills


    # ── Skill Score ────────────────────────────────────────────────────────────

    def score_skills(self, candidate: Candidate) -> tuple[float, dict]:
        cskills = self._candidate_skills_norm(candidate)

        required_hits = cskills & self.required_norm
        preferred_hits = cskills & self.preferred_norm
        nice_hits = cskills & self.nice_norm

        # Weighted coverage
        req_score = 0.0
        for s in self.required_norm:
            w = SKILL_WEIGHTS.get(s, 1.0)
            if s in cskills:
                req_score += w
        req_max = sum(SKILL_WEIGHTS.get(s, 1.0) for s in self.required_norm) or 1
        req_pct = req_score / req_max

        pref_score = len(preferred_hits) / max(len(self.preferred_norm), 1)
        nice_score = len(nice_hits) / max(len(self.nice_norm), 1)

        # Bonus for breadth — candidates with more non-overlapping skills
        breadth_bonus = min(0.1, len(cskills - self.required_norm - self.preferred_norm) * 0.005)

        total = (req_pct * 0.65) + (pref_score * 0.25) + (nice_score * 0.05) + breadth_bonus
        total = min(1.0, total)

        return total, {
            "required_coverage": round(req_pct, 3),
            "required_matched": sorted(required_hits),
            "required_missing": sorted(self.required_norm - cskills),
            "preferred_matched": sorted(preferred_hits),
            "nice_matched": sorted(nice_hits),
            "breadth_bonus": round(breadth_bonus, 3),
        }

    # ── Experience Score ───────────────────────────────────────────────────────

    def score_experience(self, candidate: Candidate) -> tuple[float, dict]:
        years = candidate.years_experience
        min_exp = self.job.min_years_experience
        max_exp = self.job.max_years_experience or 99

        # Ideal band for the seniority
        ideal_lo, ideal_hi = SENIORITY_EXPERIENCE.get(self.job.seniority, (min_exp, max_exp))

        if years < min_exp:
            # Under-qualified — penalize proportionally
            gap = min_exp - years
            score = max(0.0, 1.0 - (gap / max(min_exp, 1)) * 0.8)
        elif years > max_exp:
            # Over-qualified — mild penalty
            over = years - max_exp
            score = max(0.5, 1.0 - over * 0.03)
        elif ideal_lo <= years <= ideal_hi:
            score = 1.0
        else:
            score = 0.85  # In range but not ideal band

        # Title relevance bonus
        title_bonus = 0.0
        jt = self.job.title.lower()
        for title in candidate.previous_titles:
            tl = title.lower()
            if any(kw in tl for kw in jt.split()):
                title_bonus += 0.05
        title_bonus = min(0.15, title_bonus)

        total = min(1.0, score + title_bonus)

        return total, {
            "years_experience": years,
            "required_range": f"{min_exp}–{max_exp if max_exp < 99 else '∞'}",
            "ideal_band": f"{ideal_lo}–{ideal_hi}",
            "base_score": round(score, 3),
            "title_bonus": round(title_bonus, 3),
        }

    # ── Education Score ────────────────────────────────────────────────────────

    def score_education(self, candidate: Candidate) -> tuple[float, dict]:
        level = candidate.education_level.lower()
        requirement = self.job.education_requirement.lower()

        allowed = EDUCATION_REQUIREMENTS.get(requirement, EDUCATION_REQUIREMENTS["any"])
        base = EDUCATION_SCORES.get(level, 0.3)

        meets_req = level in allowed
        penalty = 0.0 if meets_req else 0.3

        # Field relevance heuristic
        field_bonus = 0.0
        edu_lower = candidate.education.lower()
        relevant_fields = ["computer science", "software", "data science", "mathematics",
                           "statistics", "information", "engineering", "ai", "machine learning"]
        if any(f in edu_lower for f in relevant_fields):
            field_bonus = 0.1

        total = min(1.0, max(0.0, base - penalty + field_bonus))

        return total, {
            "level": level,
            "field": candidate.education,
            "meets_requirement": meets_req,
            "field_bonus": round(field_bonus, 3),
        }

    # ── Culture Fit Score ──────────────────────────────────────────────────────

    def score_culture_fit(self, candidate: Candidate) -> tuple[float, dict]:
                signals = {}
                score = 0.5

                title_text = (
                    (candidate.current_title or "") + " " +
                    " ".join(candidate.previous_titles or [])
                ).lower()

                company_text = (
                    (candidate.current_company or "") + " " +
                    " ".join(candidate.companies or [])
                ).lower()

                industry_text = (candidate.current_industry or "").lower()

                # ML / AI titles
                if any(x in title_text for x in [
                    "machine learning",
                    "ml engineer",
                    "ai engineer",
                    "data scientist",
                    "applied scientist",
                    "nlp engineer",
                    "staff machine learning engineer",
                    "senior machine learning engineer"
                ]):
                    score += 0.20
                    signals["relevant_title"] = True

                # Strong companies
                top_companies = {
                    "google", "meta", "amazon", "microsoft",
                    "openai", "anthropic", "deepmind",
                    "databricks", "snowflake",
                    "yellow.ai", "niramai"
                }

                company_hits = [
                    c for c in company_text.split()
                    if any(t in c for t in top_companies)
                ]

                if company_hits:
                    score += 0.15
                    signals["top_company_experience"] = True

                # Relevant industry
                if any(x in industry_text for x in [
                    "ai",
                    "artificial intelligence",
                    "software",
                    "technology",
                    "data",
                    "machine learning"
                ]):
                    score += 0.15
                    signals["relevant_industry"] = True

                # Certifications
                if candidate.certifications:
                    score += min(0.10, len(candidate.certifications) * 0.03)
                    signals["certifications"] = len(candidate.certifications)

                # Open source
                if candidate.open_source:
                    score += 0.05
                    signals["open_source"] = True

                return min(1.0, score), signals

    # ── Growth Potential Score ─────────────────────────────────────────────────

    def score_growth_potential(self, candidate: Candidate) -> tuple[float, dict]:
                    score = 0.5
                    signals = {}

                    # Publications
                    if candidate.publications > 0:
                        bonus = min(0.2, candidate.publications * 0.04)
                        score += bonus
                        signals["publications"] = candidate.publications

                    # Open source contributions
                    if candidate.open_source:
                        score += 0.1
                        signals["open_source"] = True

                    # GitHub presence
                    if candidate.github_url:
                        score += 0.05
                        signals["github_profile"] = True

                    # Multilingual — signals adaptability
                    if len(candidate.languages_spoken) > 1:
                        score += 0.03
                        signals["multilingual"] = candidate.languages_spoken

                    # Career trajectory — are titles getting more senior?
                    seniority_keywords = ["junior", "associate", "mid", "senior", "lead", "staff",
                                        "principal", "architect", "director", "vp", "head"]
                    trajectory_score = 0
                    for title in candidate.previous_titles:
                        tl = title.lower()
                        for i, kw in enumerate(seniority_keywords):
                            if kw in tl:
                                trajectory_score = max(trajectory_score, i)
                    if trajectory_score >= 4:  # senior+ trajectory
                        score += 0.1
                        signals["senior_trajectory"] = True

                    total = min(1.0, score)
                    return total, signals

    
    def score_logistics(self, candidate: Candidate) -> tuple[float, dict]:
        score = 0.0
        signals = {}

        if candidate.open_to_work:
            score += 0.25
            signals["open_to_work"] = True

        if candidate.willing_to_relocate:
            score += 0.15
            signals["willing_to_relocate"] = True

        if candidate.notice_period_days <= 30:
            score += 0.20
            signals["short_notice"] = True
        elif candidate.notice_period_days <= 60:
            score += 0.10
            signals["medium_notice"] = True

        score += min(candidate.recruiter_response_rate, 1.0) * 0.20
        score += min(candidate.interview_completion_rate, 1.0) * 0.10

        total = min(1.0, score)

        return total, signals

        

    def rank(self, candidates: list[Candidate]) -> list[ScoredCandidate]:
        scored = []
        for c in candidates:
                    skill_score, skill_detail = self.score_skills(c)
                    exp_score, exp_detail = self.score_experience(c)
                    edu_score, edu_detail = self.score_education(c)
                    culture_score, culture_detail = self.score_culture_fit(c)
                    growth_score, growth_detail = self.score_growth_potential(c)
                    logis_score, logis_detail = self.score_logistics(c)

                    total = (
                        skill_score   * self.WEIGHTS["skill"] +
                        exp_score     * self.WEIGHTS["experience"] +
                        edu_score     * self.WEIGHTS["education"] +
                        culture_score * self.WEIGHTS["culture_fit"] +
                        growth_score  * self.WEIGHTS["growth_potential"] +
                        logis_score   * self.WEIGHTS["logistics"]
                    )

                    strengths, gaps = self._explain(c, skill_detail, exp_detail, edu_detail)

                    scored.append(ScoredCandidate(
                        candidate=c,
                        total_score=round(total, 4),
                        rank=0,
                        tier="",
                        skill_score=round(skill_score, 4),
                        experience_score=round(exp_score, 4),
                        education_score=round(edu_score, 4),
                        culture_fit_score=round(culture_score, 4),
                        growth_potential_score=round(growth_score, 4),
                        logistics_score=round(logis_score, 4),
                        score_breakdown={
                            "skill": skill_detail,
                            "experience": exp_detail,
                            "education": edu_detail,
                            "culture_fit": culture_detail,
                            "growth_potential": growth_detail,
                            "logistics": logis_detail,
                        },
                        strengths=strengths,
                        gaps=gaps,
                        recommendation=self._recommendation(total),
                    ))

        scored.sort(key=lambda x: x.total_score, reverse=True)

        for i, s in enumerate(scored, 1):
            s.rank = i
            s.tier = self._tier(s.total_score)

        return scored

    def _tier(self, score: float) -> str:
        if score >= 0.85: return "S"
        if score >= 0.70: return "A"
        if score >= 0.55: return "B"
        if score >= 0.40: return "C"
        return "D"

    def _recommendation(self, score: float) -> str:
        if score >= 0.85: return "Strong Hire — Fast-track to final round"
        if score >= 0.70: return "Hire — Schedule technical interview"
        if score >= 0.55: return "Consider — Phone screen recommended"
        if score >= 0.40: return "Borderline — Review manually before deciding"
        return "Pass — Does not meet minimum requirements"

    def _explain(self, c: Candidate, skill_d: dict, exp_d: dict, edu_d: dict) -> tuple[list, list]:
        strengths, gaps = [], []
        if skill_d["required_coverage"] >= 0.8:
            strengths.append(f"Strong skill match ({skill_d['required_coverage']*100:.0f}% of required skills)")
        if skill_d["required_coverage"] < 0.5:
            missing = skill_d.get("required_missing", [])[:3]
            gaps.append(f"Missing key skills: {', '.join(missing)}")
        if c.years_experience >= self.job.min_years_experience:
            strengths.append(f"{c.years_experience} years experience — meets requirement")
        else:
            gaps.append(f"Experience gap: {c.years_experience}y vs {self.job.min_years_experience}y required")
        if c.open_source:
            strengths.append("Active open-source contributor")
        if c.publications > 0:
            strengths.append(f"{c.publications} publication(s)")
        if not edu_d["meets_requirement"]:
            gaps.append(f"Education: {edu_d['level']} (role requires {self.job.education_requirement})")
        return strengths, gaps
