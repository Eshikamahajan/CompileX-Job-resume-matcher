# Tech Vriksh - AI Job Match & Skill Gap Recommender

This is a Streamlit-based resume-to-job matching application. It accepts a resume, fetches live job listings from Jooble, performs an initial TF-IDF similarity ranking, sends the best-ranked jobs to Gemini for deeper semantic evaluation, and then generates course recommendations for the missing skills in each matched role.

`app.py` is the complete runnable implementation, while `self_app.py` mirrors the same structure with TODO placeholders so someone can fill in the pipeline themselves.

## High-Level Flow

1. The user enters Jooble and Gemini API keys in the Streamlit sidebar.
2. The user chooses search keywords, location, and number of Jooble result pages to ingest.
3. The user uploads a resume in PDF or DOCX format.
4. The app extracts plain text from the resume.
5. Jooble jobs are fetched and converted into a pandas DataFrame.
6. HTML snippets from the job descriptions are cleaned.
7. TF-IDF vectorization compares the resume against each job description.
8. The top 10 jobs are selected by cosine similarity.
9. Gemini evaluates the top jobs semantically and returns structured JSON.
10. The app displays ranked job matches, fit scores, matching skills, missing skills, and rationale.
11. For jobs with missing skills, Gemini can generate an upskilling roadmap.
12. The roadmap is rendered as skill-by-skill course recommendations with search links.

## Main Application: `app.py`

`app.py` is the primary executable Streamlit application.

Run it with:

```bash
streamlit run app.py
```

From the project root, that would be:

```bash
cd tech_vriksh
streamlit run app.py
```

### Environment Loading

At startup, the app calls:

```python
load_dotenv(override=True)
```

This loads values from a local `.env` file and allows those values to override existing environment variables.

The app expects:

```env
api_key=your_jooble_api_key
GEMINI_API_KEY=your_gemini_api_key
```

The TF-IDF stage is a cheap filter. It narrows many jobs down to the most textually relevant ones before sending only the top jobs to Gemini. That reduces LLM cost, latency, and prompt size.

Important nuance:

TF-IDF is keyword/phrase driven. It can miss semantic similarity when two terms are conceptually related but lexically different. For example, a resume mentioning "PyTorch" may be relevant to a job asking for "deep learning" even if the exact wording differs. The Gemini stage handles that deeper semantic judgment.

### `evaluate_matches_with_gemini(resume_text: str, top_10_df: pd.DataFrame, client: genai.Client) -> MatchReport`

Purpose:

Asks Gemini to evaluate the top TF-IDF-ranked jobs using semantic reasoning.

Input:

- `resume_text`: cleaned resume text
- `top_10_df`: DataFrame containing the best TF-IDF matches
- `client`: initialized `google.genai.Client`

Job payload:

The function sends only these fields to Gemini:

```text
id, title, company, clean_description
```

The response is parsed directly into the `MatchReport` Pydantic model imported from `structured_output_helper.py`.

The low temperature (`0.2`) is meant to keep results more stable and less creatively varied.

### Main UI

The main page:

- Shows the app title and pipeline caption.
- Provides a file uploader for PDF and DOCX resumes.
- Parses the uploaded resume immediately.
- Shows a success message with extracted character count.
- Provides an expander with the first 1000 characters of parsed resume text.
- Runs the matching pipeline only after the user clicks the primary button.

### Pipeline Status UI

When the button is clicked, Streamlit shows a multi-step status block:

1. Fetching jobs from Jooble.
2. Running TF-IDF ranking.
3. Running Gemini semantic evaluation.
4. Marking the pipeline complete.

If either API key is missing, the app shows an error and stops execution with `st.stop()`.

Using `st.session_state` lets results persist across Streamlit reruns triggered by button clicks and UI interaction.

### Results Display

If `df_results` exists in session state, the app displays:

- Candidate summary from Gemini.
- A list of semantic job matches.
- Each job inside a bordered Streamlit container.
- Job title and company.
- Semantic fit percentage via `st.metric`.
- Gemini rationale.
- Matching skills.
- Missing skills.

If a job has missing skills, the app adds an expander where the user can generate learning recommendations.

### Course Recommendation Display

Inside each skill-gap expander:

- A button triggers `generate_upskilling_roadmap()`.
- The generated roadmap is stored in session state using a per-row key.
- Each missing skill is displayed as its own learning path.
- Each course displays:
  - level
  - course name
  - generated search link
  - provider/platform
  - key takeaway

## Mental Model

The app uses a two-stage matching strategy:

1. TF-IDF quickly answers: "Which jobs use language most similar to this resume?"
2. Gemini then answers: "Which of those jobs are actually a good semantic fit, and what is missing?"

That combination is the central design decision. TF-IDF keeps the candidate set small and cheap. Gemini adds recruiter-like interpretation, structured scoring, and personalized skill-gap recommendations.
