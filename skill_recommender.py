import urllib.parse
from pydantic import BaseModel, Field

class CourseRecommendation(BaseModel):
    level: str = Field(description="'Beginner', 'Intermediate', or 'Advanced'")
    course_name: str = Field(description="Name of the course, book, or official certification")
    platform_or_provider: str = Field(description="Platform (e.g., Coursera, Udemy, edX, O'Reilly, DeepLearning.AI)")
    key_takeaway: str = Field(description="One sentence on what the learner masters here")

class SkillLearningPath(BaseModel):
    skill_name: str = Field(description="The missing skill being bridged")
    courses: list[CourseRecommendation]

class UpskillingRoadmap(BaseModel):
    target_role: str = Field(description="Job title being targeted")
    company: str = Field(description="Company offering the role")
    learning_paths: list[SkillLearningPath]

# -------------------------------------------------------------
# Standalone Link Generator (Never fails with AttributeError)
# -------------------------------------------------------------
def get_course_url(course: CourseRecommendation) -> str:
    platform_lower = course.platform_or_provider.lower()
    encoded_name = urllib.parse.quote_plus(course.course_name)
    encoded_full = urllib.parse.quote_plus(f"{course.course_name} {course.platform_or_provider}")
    
    if "coursera" in platform_lower:
        return f"https://www.coursera.org/search?query={encoded_name}"
    elif "udemy" in platform_lower:
        return f"https://www.udemy.com/courses/search/?q={encoded_name}"
    elif "edx" in platform_lower:
        return f"https://www.edx.org/search?q={encoded_name}"
    else:
        return f"https://www.google.com/search?q={encoded_full}"