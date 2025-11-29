import streamlit as st
import pandas as pd
import re
from datetime import datetime, timedelta

# Optional PDF reader for ATS
try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# -------------------------- BASIC UTILITIES -------------------------- #

# Load CSS
def load_css():
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("styles.css not found. UI will use default Streamlit theme.")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """Small helper to load CSV files."""
    return pd.read_csv(path)


def similarity_score(text: str, keywords) -> int:
    """
    Tiny scoring helper: counts how many keywords appear in text.
    keywords can be list of words or tokens.
    """
    text = (text or "").lower()
    scores = 0
    for kw in keywords:
        if isinstance(kw, str) and kw.strip():
            if kw.lower() in text:
                scores += 1
    return scores


def extract_keywords(text: str, min_len: int = 4, top_k: int = 20):
    """Very lightweight keyword extractor based on frequency."""
    if not text:
        return []
    text = re.sub(r"[^a-zA-Z0-9\s+]", " ", text.lower())
    tokens = [t for t in text.split() if len(t) >= min_len]

    stop = {
        "this", "that", "with", "from", "have", "will", "your", "about",
        "their", "there", "where", "which", "been", "also", "into", "using",
        "some", "more", "than", "then", "them", "they", "over", "only"
    }
    tokens = [t for t in tokens if t not in stop]

    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_items[:top_k]]


# -------------------------- PAGE CONFIG & GLOBAL UI -------------------------- #

st.set_page_config(
    page_title="HR Multi-Agent Intelligence System",
    page_icon="🤖",
    layout="wide",
)

load_css()

st.markdown(
    "<div class='title-text'>HR Multi-Agent Intelligence System</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle-text'>Unified workspace for HR assistance, talent analytics, and interview intelligence.</div>",
    unsafe_allow_html=True,
)


# -------------------------- SIDEBAR NAVIGATION -------------------------- #

st.sidebar.header("Navigation")
agent = st.sidebar.radio(
    "Go to section",
    [
        "🏠 Overview",
        "👩‍💼 HR Assistant",
        "📄 Resume Screening",
        "🎤 Interview Bot",
        "🧾 Onboarding Planner",
        "📑 ATS Resume Checker",
        "🧠 Role Fit Advisor",
        "🗣 Interview Answer Analyzer",
    ],
)


st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · Streamlit · CSV data")


# -------------------------- 1. OVERVIEW / HOME -------------------------- #

if agent == "🏠 Overview":
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.subheader("Dashboard")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-title'>Policies configured</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>10+</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-title'>Candidate profiles</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>8 (sample)</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-title'>Intelligent agents</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>7</div>", unsafe_allow_html=True)

    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        st.write("### What this system can do")
        st.write("- Answer common HR policy questions\n"
                 "- Score and rank resumes for a job\n"
                 "- Run mock interviews and give feedback\n"
                 "- Generate onboarding plans for new employees\n"
                 "- Check resumes against job descriptions (ATS style)\n"
                 "- Suggest best-fit roles for a candidate\n"
                 "- Analyse interview answers for structure & clarity")

    with colB:
        st.write("### How a recruiter would use it")
        st.write("1. Use **Resume Screening** or **ATS Checker** on applicants.\n"
                 "2. Shortlisted candidates go through the **Interview Bot**.\n"
                 "3. **Role Fit Advisor** suggests where the profile fits best.\n"
                 "4. For selected candidates, use **Onboarding Planner**.\n"
                 "5. Use **Interview Answer Analyzer** to refine answers for training.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------- 2. HR ASSISTANT -------------------------- #

elif agent == "👩‍💼 HR Assistant":
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.subheader("HR Assistant – Policies, Leave & Benefits")

    data = load_csv("data/hr_policies.csv")

    query = st.text_input("Type your HR question (casual leave, notice period, insurance, etc.):")

    if st.button("Search policy"):
        if query.strip():
            scores = []
            for _, row in data.iterrows():
                text_blob = f"{row['category']} {row['question_example']} {row['answer']}"
                scores.append(similarity_score(text_blob, query.split()))
            data["score"] = scores
            best = data.sort_values("score", ascending=False).head(3)

            if best.iloc[0]["score"] == 0:
                st.warning("I couldn’t find a close match. Try using different words.")
            else:
                st.write("#### Closest matches")
                for _, row in best.iterrows():
                    st.markdown(f"**Category:** {row['category']}")
                    st.markdown(f"**Example query:** {row['question_example']}")
                    st.markdown(f"**Answer:** {row['answer']}")
                    st.markdown("---")
        else:
            st.warning("Please enter a question first.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------- 3. RESUME SCREENING -------------------------- #

elif agent == "📄 Resume Screening":
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.subheader("Resume Screening – Ranked Candidate List")

    resumes = load_csv("data/resumes.csv")
    jd = st.text_area("Paste the job description here:")

    all_skill_tokens = resumes["skills"].str.split(",").explode().dropna().unique()
    required_skills = st.multiselect(
        "Highlight these skills as mandatory (optional):",
        sorted(all_skill_tokens),
    )

    if st.button("Generate ranking"):
        if not jd.strip() and not required_skills:
            st.warning("Please provide at least a job description or select required skills.")
        else:
            scores = []
            for _, row in resumes.iterrows():
                skill_text = str(row["skills"])
                exp = float(row.get("experience_years", 0))

                score = 0
                score += similarity_score(skill_text, jd.split())
                score += 2 * similarity_score(skill_text, required_skills)
                score += exp * 1.2  # small bonus for experience
                scores.append(score)

            resumes["MatchScore"] = scores
            ranked = resumes.sort_values("MatchScore", ascending=False)

            st.write("### Ordered candidates")
            st.table(ranked[["name", "skills", "experience_years", "education", "MatchScore"]])

            st.download_button(
                label="Download ranking as CSV",
                data=ranked.to_csv(index=False),
                file_name="ranked_candidates.csv",
                mime="text/csv",
            )

            # Simple visual
            st.write("### Score overview")
            chart_df = ranked[["name", "MatchScore"]].set_index("name")
            st.bar_chart(chart_df)

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------- 4. INTERVIEW BOT -------------------------- #

elif agent == "🎤 Interview Bot":
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.subheader("Interview Practice – Question & Answer Feedback")

    qdata = load_csv("data/interview_questions.csv")
    role = st.selectbox("Select target role:", qdata["role"].unique())
    difficulty = st.radio("Pick difficulty level:", ["Easy", "Medium", "Hard"], horizontal=True)

    filtered = qdata[(qdata["role"] == role) & (qdata["difficulty"] == difficulty)]
    if filtered.empty:
        st.info("No questions for the selected combination yet.")
    else:
        row = filtered.sample(1).iloc[0]
        st.markdown(f"**Question:** {row['question']}")
        answer = st.text_area("Write your answer as you would say it in an interview:")

        if st.button("Evaluate my answer"):
            keywords = [k.strip() for k in str(row["ideal_answer_keywords"]).split(",") if k.strip()]
            score = similarity_score(answer, keywords)
            total = len(keywords)
            percent = int((score / total) * 100) if total > 0 else 0

            st.info(f"Estimated coverage: **{percent}%**  (matched {score} out of {total} key ideas)")

            st.write("Key points the evaluator expected:")
            st.markdown(", ".join(f"`{k}`" for k in keywords))

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------- 5. ONBOARDING PLANNER -------------------------- #

elif agent == "🧾 Onboarding Planner":
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.subheader("Onboarding Plan Generator")

    user_name = st.text_input("Employee name:")
    role = st.text_input("Role / designation:")
    plan = load_csv("data/onboarding_tasks.csv")
    dept = st.selectbox("Department:", plan["department"].unique())
    start_date = st.date_input("Start date", datetime.today())

    if st.button("Create onboarding schedule"):
        if not user_name.strip():
            st.warning("Please enter the employee name.")
        else:
            dplan = plan[plan["department"] == dept].copy()
            dplan["Date"] = [
                (start_date + timedelta(days=int(offset))).strftime("%d-%m-%Y")
                for offset in dplan["day_offset"]
            ]

            st.success(f"Generated plan for {user_name} – {role}")
            st.table(dplan[["Date", "task_name", "description"]])

            st.download_button(
                label="Download onboarding plan",
                data=dplan.to_csv(index=False),
                file_name=f"{user_name}_onboarding_plan.csv",
                mime="text/csv",
            )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------- 6. ATS RESUME CHECKER -------------------------- #

elif agent == "📑 ATS Resume Checker":
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.subheader("ATS-Style Resume vs JD Check")

    if pdfplumber is None:
        st.error("pdfplumber is not installed. Install it with:\n\n`python -m pip install pdfplumber`")
    else:
        uploaded_resume = st.file_uploader("Upload resume as PDF", type=["pdf"])
        jd_text = st.text_area("Paste the job description here:")

        if st.button("Analyse resume"):
            if uploaded_resume is None or not jd_text.strip():
                st.warning("Upload a resume and enter a job description first.")
            else:
                # Pull text from PDF
                resume_text = ""
                with pdfplumber.open(uploaded_resume) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        resume_text += page_text + " "

                jd_keywords = extract_keywords(jd_text, min_len=4, top_k=30)
                resume_lower = resume_text.lower()

                matched = sum(1 for kw in jd_keywords if kw in resume_lower)
                total = len(jd_keywords)
                score = int((matched / total) * 100) if total > 0 else 0

                missing = [kw for kw in jd_keywords if kw not in resume_lower][:12]

                st.progress(score / 100)
                st.markdown(f"**ATS-style keyword score:** `{score}%`")

                if score >= 75:
                    st.success("Looks quite friendly to a keyword-based screening system.")
                elif score >= 50:
                    st.info("Decent start. Adding more relevant terms from the JD can help.")
                else:
                    st.warning("Many important phrases from the JD are missing in the resume.")

                st.write("### Missing or weakly represented terms")
                if missing:
                    st.markdown(", ".join(f"`{m}`" for m in missing))
                else:
                    st.success("Most important JD phrases are already present.")

                chart_df = pd.DataFrame(
                    {"Category": ["Matched", "Missed"],
                     "Count": [matched, max(total - matched, 0)]}
                ).set_index("Category")
                st.write("### Quick visual")
                st.bar_chart(chart_df)

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------- 7. ROLE FIT ADVISOR -------------------------- #

elif agent == "🧠 Role Fit Advisor":
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.subheader("Role Fit Advisor – Suggest Suitable Positions")

    st.write("Paste a candidate summary or resume text below. The system will estimate how well it fits some sample roles.")

    resume_text_input = st.text_area("Candidate profile / resume (plain text):")

    # Simple built-in role profiles (can be moved to CSV later)
    role_profiles = [
        {
            "role": "Python Backend Developer",
            "keywords": ["python", "django", "flask", "rest", "api", "database", "sql", "postgre", "backend"],
        },
        {
            "role": "Data Analyst",
            "keywords": ["excel", "sql", "power bi", "tableau", "dashboard", "analysis", "report", "visualization"],
        },
        {
            "role": "Machine Learning Engineer",
            "keywords": ["machine learning", "deep learning", "pytorch", "tensorflow", "model", "training", "prediction"],
        },
        {
            "role": "HR Executive",
            "keywords": ["recruitment", "onboarding", "payroll", "hrms", "employee engagement", "policy", "leave"],
        },
        {
            "role": "Frontend Developer",
            "keywords": ["javascript", "react", "html", "css", "ui", "frontend", "typescript"],
        },
    ]

    if st.button("Check suitable roles"):
        if not resume_text_input.strip():
            st.warning("Please paste some resume text first.")
        else:
            results = []
            for prof in role_profiles:
                score = similarity_score(resume_text_input, prof["keywords"])
                results.append((prof["role"], score))

            # Sort by score
            results = sorted(results, key=lambda x: x[1], reverse=True)

            st.write("### Suggested matches")
            for role_name, s in results:
                if s == 0:
                    continue
                st.markdown(f"**{role_name}** – score `{s}`")
            if all(s == 0 for _, s in results):
                st.info("No strong match found with the current sample roles. Try adding more technical or HR details.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------- 8. INTERVIEW ANSWER ANALYZER -------------------------- #

elif agent == "🗣 Interview Answer Analyzer":
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    st.subheader("Interview Answer Analyzer – Structure & Clarity Check")

    st.write(
        "Paste a spoken-style answer (for example: *Tell me about yourself*, "
        "*Why should we hire you?*). The tool will estimate clarity and filler usage."
    )

    spoken_answer = st.text_area("Paste your answer text here:")

    if st.button("Analyse answer"):
        if not spoken_answer.strip():
            st.warning("Please paste an answer first.")
        else:
            text = spoken_answer.lower()
            words = [w for w in re.findall(r"\b\w+\b", text)]
            length = len(words)

            filler_words = ["um", "uh", "like", "you know", "actually", "basically", "sort of", "kind of"]
            filler_count = 0
            for f in filler_words:
                # simple count
                filler_count += text.count(f)

            filler_rate = (filler_count / length * 100) if length > 0 else 0

            positive_markers = ["led", "improved", "built", "created", "designed", "implemented", "optimised", "mentored"]
            pos_hits = similarity_score(text, positive_markers)

            # rough scoring
            structure_score = max(0, min(100, (length / 30) * 20 + pos_hits * 10 - filler_rate * 0.5))
            clarity_score = max(0, min(100, 90 - filler_rate * 1.1 + pos_hits * 3))

            st.write("### Quick summary")
            st.write(f"- Approx. word count: **{length}**")
            st.write(f"- Estimated fillers used: **{filler_count}**")
            st.write(f"- Filler density: **{filler_rate:.1f}%** of the answer")

            st.write("### Scores (rough estimate)")
            score_df = pd.DataFrame(
                {"Aspect": ["Structure / content", "Clarity of expression"],
                 "Score": [round(structure_score), round(clarity_score)]}
            ).set_index("Aspect")
            st.bar_chart(score_df)

            if filler_rate > 8:
                st.warning("Try to reduce filler words and add small pauses instead.")
            if pos_hits == 0:
                st.info("Consider adding action verbs like *built, improved, designed* to sound more impactful.")
            if length < 40:
                st.info("The answer looks a bit short. In an interview, 60–90 seconds is a good target for such questions.")

    st.markdown("</div>", unsafe_allow_html=True)
