"""
Unit + integration tests for the NeuralRank system
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Minimal pytest stub so tests still run without it
    class _pytest_stub:
        @staticmethod
        def fixture(fn=None, **kwargs):
            if fn: return fn
            return lambda f: f
    pytest = _pytest_stub()

from src.ranker import (
    Candidate, JobDescription, CandidateRanker,
    EDUCATION_SCORES, SENIORITY_EXPERIENCE
)
from src.data import generate_dataset, sample_job_description


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def job():
    return sample_job_description()

@pytest.fixture
def ranker(job):
    return CandidateRanker(job)

def make_candidate(**kwargs) -> Candidate:
    defaults = dict(
        id="TEST-001", name="Test User", email="test@email.com",
        years_experience=7.0,
        skills=["Python","Machine Learning","Deep Learning","Large Language Models","RAG","PyTorch","MLOps","AWS"],
        education="MSc Computer Science", education_level="masters",
        previous_titles=["Senior ML Engineer"],
        companies=["Google"],
        location="San Francisco, CA", remote_preference="hybrid",
        salary_expectation=160000,
        achievements=["Reduced latency by 40%"],
        certifications=["AWS Certified ML Specialty"],
        open_source=True, publications=2,
        languages_spoken=["English"],
        linkedin_url=None, github_url="https://github.com/testuser",
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


# ── Skill Tests ────────────────────────────────────────────────────────────────

class TestSkillScoring:
    def test_perfect_skill_match(self, ranker):
        c = make_candidate(skills=["Python","Machine Learning","Deep Learning",
                                    "Large Language Models","Retrieval Augmented Generation",
                                    "PyTorch","MLOps","Amazon Web Services"])
        score, detail = ranker.score_skills(c)
        assert detail["required_coverage"] >= 0.99, f"Expected full coverage, got {detail['required_coverage']}"
        assert score >= 0.60, f"Expected >=0.60, got {score}"

    def test_no_skills_match(self, ranker):
        c = make_candidate(skills=["Java","Ruby","PHP"])
        score, detail = ranker.score_skills(c)
        assert score < 0.20, f"Expected <0.20, got {score}"

    def test_synonym_matching(self, ranker):
        # "LLMs" should match "Large Language Models"
        c_with_abbrev = make_candidate(skills=["Python","LLMs","PyTorch","RAG","MLOps","AWS",
                                                "Machine Learning","Deep Learning"])
        c_with_full = make_candidate(skills=["Python","Large Language Models","PyTorch","RAG",
                                              "MLOps","AWS","Machine Learning","Deep Learning"])
        s1, _ = ranker.score_skills(c_with_abbrev)
        s2, _ = ranker.score_skills(c_with_full)
        assert abs(s1 - s2) < 0.05, f"Synonym mismatch: {s1} vs {s2}"

    def test_preferred_skills_add_bonus(self, ranker):
        c_no_pref = make_candidate(skills=["Python","Machine Learning","Deep Learning",
                                            "Large Language Models","RAG","PyTorch","MLOps","AWS"])
        c_with_pref = make_candidate(skills=["Python","Machine Learning","Deep Learning",
                                              "Large Language Models","RAG","PyTorch","MLOps","AWS",
                                              "Kubernetes","Docker","LangChain"])
        s1, _ = ranker.score_skills(c_no_pref)
        s2, _ = ranker.score_skills(c_with_pref)
        assert s2 >= s1

    def test_partial_skill_match(self, ranker):
        c = make_candidate(skills=["Python","Machine Learning"])
        score, detail = ranker.score_skills(c)
        assert 0.0 < score < 0.9

    def test_case_insensitive_matching(self, ranker):
        c_lower = make_candidate(skills=["python","machine learning","deep learning",
                                          "large language models","rag","pytorch","mlops","aws"])
        c_upper = make_candidate(skills=["Python","Machine Learning","Deep Learning",
                                          "Large Language Models","RAG","PyTorch","MLOps","AWS"])
        s1, _ = ranker.score_skills(c_lower)
        s2, _ = ranker.score_skills(c_upper)
        assert abs(s1 - s2) < 0.05


# ── Experience Tests ───────────────────────────────────────────────────────────

class TestExperienceScoring:
    def test_ideal_experience(self, ranker):
        c = make_candidate(years_experience=8.0)
        score, _ = ranker.score_experience(c)
        assert score >= 0.90

    def test_under_experience(self, ranker):
        c = make_candidate(years_experience=2.0)  # min is 5
        score, _ = ranker.score_experience(c)
        assert score < 0.7

    def test_over_experience_mild_penalty(self, ranker):
        c = make_candidate(years_experience=18.0)  # max is 15
        score, _ = ranker.score_experience(c)
        assert 0.5 <= score <= 1.0  # Should not be zero

    def test_title_bonus(self, ranker):
        c_relevant = make_candidate(previous_titles=["Senior ML Engineer","ML Engineer"])
        c_irrelevant = make_candidate(previous_titles=["Frontend Developer","QA Analyst"])
        s1, _ = ranker.score_experience(c_relevant)
        s2, _ = ranker.score_experience(c_irrelevant)
        assert s1 >= s2


# ── Education Tests ────────────────────────────────────────────────────────────

class TestEducationScoring:
    def test_phd_highest(self, ranker):
        c = make_candidate(education_level="phd", education="PhD Computer Science")
        score, _ = ranker.score_education(c)
        assert score >= 0.9

    def test_meets_requirement(self, ranker):
        c_bachelors = make_candidate(education_level="bachelors", education="BSc CS")
        score, detail = ranker.score_education(c_bachelors)
        assert detail["meets_requirement"] is True
        assert score > 0.5

    def test_below_requirement_penalty(self):
        job_strict = JobDescription(
            title="Research Scientist", company="Lab",
            required_skills=["Python"], preferred_skills=[],
            min_years_experience=5, max_years_experience=None,
            education_requirement="phd",
            location="NYC", remote_policy="hybrid", salary_range=None,
            key_responsibilities=[], nice_to_haves=[], seniority="senior",
        )
        ranker_strict = CandidateRanker(job_strict)
        c = make_candidate(education_level="bachelors", education="BSc CS")
        score, detail = ranker_strict.score_education(c)
        assert detail["meets_requirement"] is False
        assert score < 0.6

    def test_relevant_field_bonus(self, ranker):
        c_relevant = make_candidate(education="MSc Machine Learning", education_level="masters")
        c_irrelevant = make_candidate(education="MSc History", education_level="masters")
        s1, _ = ranker.score_education(c_relevant)
        s2, _ = ranker.score_education(c_irrelevant)
        assert s1 > s2


# ── Ranking Integration Tests ──────────────────────────────────────────────────

class TestRanking:
    def test_better_candidate_ranks_higher(self, ranker):
        strong = make_candidate(
            id="STRONG",
            skills=["Python","Machine Learning","Deep Learning","Large Language Models",
                    "RAG","PyTorch","MLOps","AWS","Kubernetes","Docker"],
            years_experience=8.0, education_level="phd",
            open_source=True, publications=3,
        )
        weak = make_candidate(
            id="WEAK",
            skills=["Java","SQL"],
            years_experience=1.0, education_level="bootcamp",
            open_source=False, publications=0,
        )
        ranked = ranker.rank([weak, strong])
        assert ranked[0].candidate.id == "STRONG"
        assert ranked[1].candidate.id == "WEAK"

    def test_ranks_are_sequential(self, ranker):
        candidates = generate_dataset(n=20, seed=99)
        ranked = ranker.rank(candidates)
        for i, s in enumerate(ranked, 1):
            assert s.rank == i

    def test_scores_in_range(self, ranker):
        candidates = generate_dataset(n=50, seed=7)
        ranked = ranker.rank(candidates)
        for s in ranked:
            assert 0.0 <= s.total_score <= 1.0
            assert 0.0 <= s.skill_score <= 1.0
            assert 0.0 <= s.experience_score <= 1.0

    def test_tier_assignment(self, ranker):
        candidates = generate_dataset(n=100, seed=1)
        ranked = ranker.rank(candidates)
        for s in ranked:
            assert s.tier in ("S","A","B","C","D")

    def test_strengths_and_gaps_populated(self, ranker):
        candidates = generate_dataset(n=10, seed=5)
        ranked = ranker.rank(candidates)
        for s in ranked:
            assert isinstance(s.strengths, list)
            assert isinstance(s.gaps, list)

    def test_large_dataset_performance(self, ranker):
        import time
        candidates = generate_dataset(n=1000, seed=42)
        t0 = time.time()
        ranked = ranker.rank(candidates)
        elapsed = time.time() - t0
        assert elapsed < 10.0, f"Ranking 1000 candidates took {elapsed:.2f}s (too slow)"
        assert len(ranked) == 1000

    def test_reproducible_results(self, ranker):
        candidates = generate_dataset(n=50, seed=42)
        r1 = ranker.rank(candidates)
        r2 = ranker.rank(candidates)
        for a, b in zip(r1, r2):
            assert a.total_score == b.total_score
            assert a.rank == b.rank


# ── Data Generation Tests ──────────────────────────────────────────────────────

class TestDataGeneration:
    def test_generates_correct_count(self):
        candidates = generate_dataset(n=100)
        assert len(candidates) == 100

    def test_unique_ids(self):
        candidates = generate_dataset(n=50)
        ids = [c.id for c in candidates]
        assert len(ids) == len(set(ids))

    def test_reproducible_with_seed(self):
        c1 = generate_dataset(n=5, seed=42)
        c2 = generate_dataset(n=5, seed=42)
        for a, b in zip(c1, c2):
            assert a.id == b.id
            assert a.name == b.name
            assert a.skills == b.skills


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
