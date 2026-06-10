"""
Data layer: sample dataset generation + loader utilities
"""

import json
import random
from pathlib import Path
from src.ranker import Candidate, JobDescription


# ─── Sample Data Generation ──────────────────────────────────────────────────────

FIRST_NAMES = ["Alex","Jordan","Morgan","Taylor","Riley","Casey","Quinn","Avery","Skylar","Parker",
               "Drew","Cameron","Blake","Reese","Peyton","Sage","Rowan","Charlie","Finley","River",
               "Priya","Arjun","Chen","Wei","Aisha","Omar","Leila","Yuki","Soren","Nadia"]
LAST_NAMES = ["Chen","Patel","Rodriguez","Kim","Johnson","Williams","Brown","Taylor","Anderson",
              "Thompson","Garcia","Martinez","Robinson","Lewis","Lee","Walker","Hall","Allen","Young",
              "Hernandez","King","Wright","Scott","Torres","Nguyen","Hill","Flores","Green","Adams","Nelson"]

COMPANIES = [
    "Google","Meta","Amazon","Microsoft","Apple","OpenAI","Anthropic","DeepMind","Netflix","Stripe",
    "Airbnb","Uber","Lyft","Databricks","Snowflake","Palantir","Figma","Notion","Vercel","Hugging Face",
    "Scale AI","Cohere","Mistral","Stability AI","Weights & Biases","DataRobot","H2O.ai","C3.ai",
    "TechCorp Inc","DataFlow Systems","InnovateSoft","CloudBridge","NeuralPath","VectorDB Ltd",
    "StartupXYZ","FinTech Co","HealthAI","EduTech Labs","RetailBot","AutomationWorks",
]

SKILL_POOL = {
    "ml_core": ["Python","Machine Learning","Deep Learning","TensorFlow","PyTorch","Scikit-learn",
                "XGBoost","LightGBM","Keras","JAX"],
    "llm": ["Large Language Models","RAG","Fine-tuning","Prompt Engineering","LangChain","LlamaIndex",
            "OpenAI API","Hugging Face Transformers","RLHF","Vector Databases"],
    "mlops": ["MLOps","Kubernetes","Docker","Airflow","MLflow","Kubeflow","DVC","Feature Stores",
              "Model Monitoring","CI/CD"],
    "data": ["SQL","Pandas","Spark","Databricks","dbt","Snowflake","BigQuery","Data Pipelines",
             "ETL","Data Modeling"],
    "cloud": ["AWS","GCP","Azure","Lambda","S3","ECS","Vertex AI","SageMaker","Azure ML"],
    "general": ["Git","Linux","REST APIs","GraphQL","Microservices","System Design","Technical Writing",
                "Agile","Code Review","Mentoring"],
}

TITLES = {
    "junior": ["Junior ML Engineer","Data Analyst","Software Engineer I","ML Research Intern","Data Scientist I"],
    "mid": ["ML Engineer","Data Scientist","Software Engineer II","AI Engineer","NLP Engineer"],
    "senior": ["Senior ML Engineer","Senior Data Scientist","Senior AI Engineer","ML Tech Lead","Staff Data Scientist"],
    "lead": ["Lead ML Engineer","Principal Data Scientist","ML Architect","Head of ML Platform","AI Research Lead"],
}

EDUCATIONS = [
    ("PhD Computer Science","phd"),("PhD Machine Learning","phd"),("PhD Statistics","phd"),
    ("MSc Computer Science","masters"),("MSc Data Science","masters"),("MSc AI","masters"),
    ("BSc Computer Science","bachelors"),("BSc Software Engineering","bachelors"),("BSc Mathematics","bachelors"),
    ("BSc Statistics","bachelors"),("BSc Information Technology","bachelors"),
    ("Bootcamp: Data Science","bootcamp"),("Bootcamp: Full Stack","bootcamp"),
    ("Self-taught","self-taught"),("Associate Degree in CS","associate"),
]

ACHIEVEMENTS = [
    "Reduced model inference latency by 40% using quantization",
    "Increased model accuracy from 87% to 94% on production dataset",
    "Led a team of 5 engineers to deliver ML platform on schedule",
    "Built RAG pipeline serving 50K+ daily queries with 99.9% uptime",
    "Automated data pipeline reducing processing time from 6h to 20min",
    "Published 3 papers at NeurIPS and ICML",
    "Launched recommender system generating $2M additional revenue",
    "Scaled ML infrastructure to handle 10x traffic growth",
    "Open-sourced ML toolkit with 2k+ GitHub stars",
    "Mentored 8 junior engineers across 2 teams",
    "Reduced cloud costs by 35% through model optimization",
    "Delivered real-time fraud detection model with <50ms latency",
    "Grew ML team from 3 to 12 engineers",
    "Built LLM-powered feature reducing customer support tickets by 60%",
    "Implemented feature store that improved model training speed by 3x",
]

CERTIFICATIONS = [
    "AWS Certified ML Specialty","Google Cloud Professional ML Engineer",
    "TensorFlow Developer Certificate","Databricks Certified ML Professional",
    "Azure AI Engineer Associate","Deep Learning Specialization (Coursera)",
    "MLOps Specialization (Coursera)","Stanford ML Certificate",
]


def random_candidate(candidate_id: int, seed: int = None) -> Candidate:
    if seed is not None:
        random.seed(seed + candidate_id * 37)

    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"

    # Pick career level
    level = random.choices(
        ["junior", "mid", "senior", "lead"],
        weights=[20, 35, 35, 10]
    )[0]

    years_map = {"junior": (0.5, 3), "mid": (3, 7), "senior": (6, 14), "lead": (10, 22)}
    lo, hi = years_map[level]
    years_exp = round(random.uniform(lo, hi), 1)

    # Build skills — weighted toward ML domain
    skill_buckets = list(SKILL_POOL.keys())
    selected_skills = set()
    for bucket in ["ml_core", "llm", "mlops", "data", "cloud", "general"]:
        n = random.randint(1, min(5, len(SKILL_POOL[bucket])))
        selected_skills.update(random.sample(SKILL_POOL[bucket], n))
    # Junior candidates have fewer skills
    if level == "junior":
        selected_skills = set(random.sample(list(selected_skills), max(4, len(selected_skills)//2)))

    # Education
    edu_weights = {"junior": [2,2,4,6,2,3], "mid": [3,5,6,5,1,1], "senior": [6,8,7,2,1,1], "lead": [8,7,5,1,1,1]}
    edu_choices = EDUCATIONS[:3] + EDUCATIONS[3:5] + EDUCATIONS[5:7] + [EDUCATIONS[7]]
    # Simplify — just pick weighted
    edu_pool = (EDUCATIONS[:3] * 2 if level in ["senior","lead"] else []) + EDUCATIONS
    edu_tuple = random.choice(edu_pool)
    edu_name, edu_level = edu_tuple

    # Titles — career history
    num_jobs = random.randint(1, 4)
    level_pool = TITLES[level]
    if level in ["senior", "lead"]:
        level_pool = TITLES["mid"] + TITLES["senior"] + (TITLES["lead"] if level == "lead" else [])
    titles = [random.choice(level_pool) for _ in range(num_jobs)]

    companies = random.sample(COMPANIES, min(num_jobs, len(COMPANIES)))

    achievements = random.sample(ACHIEVEMENTS, random.randint(1, 5))
    certs = random.sample(CERTIFICATIONS, random.randint(0, 3))

    remote_opts = ["remote", "hybrid", "onsite", "flexible"]
    remote = random.choices(remote_opts, weights=[40, 30, 15, 15])[0]

    salary_base = {"junior": 70000, "mid": 110000, "senior": 150000, "lead": 190000}
    salary = salary_base[level] + random.randint(-10000, 30000) if random.random() > 0.2 else None

    languages = ["English"]
    if random.random() > 0.6:
        extra = random.choice(["Spanish","Mandarin","Hindi","French","German","Arabic","Japanese","Portuguese"])
        languages.append(extra)

    locations = ["San Francisco, CA","New York, NY","Austin, TX","Seattle, WA","Boston, MA",
                 "Remote","London, UK","Toronto, Canada","Berlin, Germany","Bangalore, India"]

    return Candidate(
        id=f"CAND-{candidate_id:04d}",
        name=name,
        email=f"{first.lower()}.{last.lower()}@email.com",
        years_experience=years_exp,
        skills=sorted(selected_skills),
        education=edu_name,
        education_level=edu_level,
        previous_titles=titles,
        companies=companies,
        location=random.choice(locations),
        remote_preference=remote,
        salary_expectation=salary,
        achievements=achievements,
        certifications=certs,
        open_source=random.random() > 0.65,
        publications=random.choices([0, 1, 2, 3, 5], weights=[60, 20, 10, 7, 3])[0],
        languages_spoken=languages,
        linkedin_url=f"https://linkedin.com/in/{first.lower()}-{last.lower()}-{candidate_id}",
        github_url=f"https://github.com/{first.lower()}{last.lower()}{candidate_id}" if random.random() > 0.35 else None,
        raw={},
    )


def generate_dataset(n: int = 500, seed: int = 42) -> list[Candidate]:
    random.seed(seed)
    return [random_candidate(i, seed) for i in range(1, n + 1)]


def sample_job_description() -> JobDescription:
    return JobDescription(
        title="Senior ML Engineer",
        company="NeuralPath Technologies",
        required_skills=[
            "Python", "Machine Learning", "Deep Learning", "Large Language Models",
            "RAG", "PyTorch", "MLOps", "AWS",
        ],
        preferred_skills=[
            "Kubernetes", "Docker", "Fine-tuning", "LangChain",
            "Hugging Face Transformers", "Feature Stores",
        ],
        min_years_experience=5,
        max_years_experience=15,
        education_requirement="bachelors",
        location="San Francisco, CA",
        remote_policy="hybrid",
        salary_range=(140000, 200000),
        key_responsibilities=[
            "Design and deploy production LLM-based systems",
            "Build and maintain ML pipelines and feature stores",
            "Lead architecture decisions for ML platform",
            "Mentor junior engineers",
            "Collaborate with product and research teams",
        ],
        nice_to_haves=[
            "Publications at NeurIPS/ICML/ACL", "Open source contributions",
            "Experience with RLHF", "Technical writing",
        ],
        seniority="senior",
    )


def load_candidates_from_json(path: str) -> list[Candidate]:
    with open(path) as f:
        data = json.load(f)
    candidates = []
    for raw in data:
        candidates.append(Candidate(
            id=raw.get("id", f"CAND-{len(candidates):04d}"),
            name=raw.get("name", "Unknown"),
            email=raw.get("email", ""),
            years_experience=float(raw.get("years_experience", 0)),
            skills=raw.get("skills", []),
            education=raw.get("education", ""),
            education_level=raw.get("education_level", "other"),
            previous_titles=raw.get("previous_titles", []),
            companies=raw.get("companies", []),
            location=raw.get("location", ""),
            remote_preference=raw.get("remote_preference", "flexible"),
            salary_expectation=raw.get("salary_expectation"),
            achievements=raw.get("achievements", []),
            certifications=raw.get("certifications", []),
            open_source=raw.get("open_source", False),
            publications=raw.get("publications", 0),
            languages_spoken=raw.get("languages_spoken", ["English"]),
            linkedin_url=raw.get("linkedin_url"),
            github_url=raw.get("github_url"),
            raw=raw,
        ))
    return candidates


def load_job_from_json(path: str) -> JobDescription:
    with open(path) as f:
        raw = json.load(f)
    return JobDescription(
        title=raw.get("title", ""),
        company=raw.get("company", ""),
        required_skills=raw.get("required_skills", []),
        preferred_skills=raw.get("preferred_skills", []),
        min_years_experience=float(raw.get("min_years_experience", 0)),
        max_years_experience=raw.get("max_years_experience"),
        education_requirement=raw.get("education_requirement", "any"),
        location=raw.get("location", ""),
        remote_policy=raw.get("remote_policy", "flexible"),
        salary_range=tuple(raw.get("salary_range", [None, None])) if raw.get("salary_range") else None,
        key_responsibilities=raw.get("key_responsibilities", []),
        nice_to_haves=raw.get("nice_to_haves", []),
        seniority=raw.get("seniority", "mid"),
    )
def load_candidates_jsonl(path: str):
    import json
    from src.ranker import Candidate

    candidates = []

    with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)

                profile = raw.get("profile", {})
                signals = raw.get("redrob_signals", {})

                degree = (
                    raw.get("education", [{}])[0].get("degree", "")
                    if raw.get("education")
                    else ""
                )

                degree_lower = degree.lower()

                if "phd" in degree_lower or "ph.d" in degree_lower:
                    education_level = "phd"
                elif "master" in degree_lower or "m.tech" in degree_lower:
                    education_level = "masters"
                elif "b.e" in degree_lower or "b.tech" in degree_lower:
                    education_level = "bachelors"
                else:
                    education_level = "other"

                candidates.append(
                    Candidate(
                        id=raw.get("candidate_id", ""),
                        name=profile.get("anonymized_name", "Unknown"),
                        email="",
                        years_experience=float(profile.get("years_of_experience", 0)),
                        skills=[s.get("name", "") for s in raw.get("skills", [])],

                        education=degree,
                        education_level=education_level,

                        previous_titles=[
                            r.get("title", "")
                            for r in raw.get("career_history", [])
                        ],

                        companies=[
                            r.get("company", "")
                            for r in raw.get("career_history", [])
                        ],
                    location=profile.get("location", ""),
                    remote_preference="flexible",
                    salary_expectation=None,
                    achievements=[],
                    certifications=[],
                    open_source=False,
                    publications=0,
                    languages_spoken=["English"],
                    linkedin_url=None,
                    github_url=None,

                    headline=profile.get("headline", ""),
                    summary=profile.get("summary", ""),
                    current_title=profile.get("current_title", ""),
                    current_company=profile.get("current_company", ""),
                    current_industry=profile.get("current_industry", ""),

                    recruiter_response_rate=float(
                        signals.get("recruiter_response_rate", 0)
                    ),
                    interview_completion_rate=float(
                        signals.get("interview_completion_rate", 0)
                    ),
                    github_activity_score=float(
                        signals.get("github_activity_score", 0)
                    ),
                    saved_by_recruiters_30d=int(
                        signals.get("saved_by_recruiters_30d", 0)
                    ),

                    open_to_work=bool(
                        signals.get("open_to_work_flag", False)
                    ),
                    willing_to_relocate=bool(
                        signals.get("willing_to_relocate", False)
                    ),
                    notice_period_days=int(
                        signals.get("notice_period_days", 90)
                    ),

                    raw=raw,
                )
            )

    return candidates