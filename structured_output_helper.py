from pydantic import BaseModel, Field


# 1. Define the schema for a single evaluated job match
class JobEvaluation(BaseModel):
    job_id: str = Field(description="Unique ID of the job listing")
    title: str = Field(description="Title of the job")
    company: str = Field(description="Company offering the role")
    semantic_fit_score: int = Field(description="Realistic match score from 0 to 100 based on true skills overlap")
    matching_skills: list[str] = Field(description="Key skills found in both the resume and the job requirements")
    missing_skills: list[str] = Field(description="Crucial skills or requirements mentioned in the job that the resume lacks")
    rationale: str = Field(description="A concise, 2-sentence explanation of why this job is or isn't a strong fit")

# 2. Define the container for the batch report
class MatchReport(BaseModel):
    summary: str = Field(description="One-line summary of the candidate's core profile strengths")
    evaluated_jobs: list[JobEvaluation]