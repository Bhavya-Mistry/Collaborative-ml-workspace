def build_ats_prompt(resume_text, job_role="Software Engineer"):
    return f"""
You are an ATS (Applicant Tracking System) Resume Evaluator.

Job Role: {job_role}

Below is a candidate's resume:
------------------------------
{resume_text}
------------------------------

1. Summarize the candidate's background in 2–3 lines.
2. Rate the resume's relevance to the role of {job_role} on a scale of 1 to 10.
3. Suggest 1 improvement to better match this role.

Return the result in this format:
- Summary:
- Score (out of 10):
- Suggestions:
"""