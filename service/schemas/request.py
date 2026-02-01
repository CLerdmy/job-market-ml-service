from pydantic import BaseModel


class JobFeatures(BaseModel):
    job_title: str
    company: str
    location: str
    work_type: str
    category: str
    experience: int
    skills: str
