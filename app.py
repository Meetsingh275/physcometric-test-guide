
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from career_info import career_info

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Career Prediction System",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Career Prediction System")
st.write("Find your best career based on your psychometric assessment.")

# -----------------------------
# Load Model
# -----------------------------
try:
    with open("career_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📝 Enter Your Scores")

logical = st.slider("🧠 Logical Score", 0, 100, 50)
numerical = st.slider("🔢 Numerical Score", 0, 100, 50)
communication = st.slider("🗣️ Communication Score", 0, 100, 50)
creativity = st.slider("🎨 Creativity Score", 0, 100, 50)
leadership = st.slider("👑 Leadership Score", 0, 100, 50)
technical = st.slider("💻 Technical Interest", 0, 100, 50)
helping = st.slider("🤝 Helping Nature", 0, 100, 50)
business = st.slider("💼 Business Interest", 0, 100, 50)
artistic = st.slider("🖌️ Artistic Interest", 0, 100, 50)
stress = st.slider("😌 Stress Handling", 0, 100, 50)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🚀 Predict Career"):

    sample = pd.DataFrame(
        [[
            logical,
            numerical,
            communication,
            creativity,
            leadership,
            technical,
            helping,
            business,
            artistic,
            stress
        ]],
        columns=[
            "LogicalScore",
            "NumericalScore",
            "CommunicationScore",
            "CreativityScore",
            "LeadershipScore",
            "TechnicalInterest",
            "HelpingNature",
            "BusinessInterest",
            "ArtisticInterest",
            "StressHandling"
        ]
    )

    try:

        probabilities = model.predict_proba(sample)[0]
        careers = model.classes_

        results = sorted(
            zip(careers, probabilities),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        st.divider()

        st.subheader("🏆 Top 3 Career Recommendations")

        emojis = {
            "Software Developer": "💻",
            "Data Analyst": "📊",
            "Graphic Designer": "🎨",
            "Doctor": "🩺",
            "Teacher": "📚",
            "Engineer": "⚙️",
            "Psychologist": "🧠",
            "Civil Services": "🏛️",
            "Marketing Professional": "📢",
            "Entrepreneur": "🚀"
        }

        # -----------------------------
        # Top 3 Careers
        # -----------------------------
        for i, (career, prob) in enumerate(results, start=1):

            score = prob * 100
            emoji = emojis.get(career, "⭐")

            st.subheader(f"{i}. {emoji} {career}")
            st.write(f"⭐ Match Score: {score:.2f}%")
            st.progress(float(prob))
            st.divider()

        # -----------------------------
        # Best Career
        # -----------------------------
        best_career = results[0][0]
        best_score = results[0][1] * 100

        st.success(
            f"🏆 Best Career Match: {best_career} ({best_score:.2f}%)"
        )

        st.metric(
            "🎯 Prediction Confidence",
            f"{best_score:.2f}%"
        )

        # -----------------------------
        # Skills & Roadmap
        # -----------------------------
        if best_career in career_info:

            st.subheader("📚 Recommended Skills")

            for skill in career_info[best_career]["skills"]:
                st.write(f"✅ {skill}")

            st.subheader("🛣️ Learning Roadmap")

            for step in career_info[best_career]["roadmap"]:
                st.write(f"➡️ {step}")

        # -----------------------------
        # Strong & Weak Skills
        # -----------------------------
        scores = {
            "Logical": logical,
            "Numerical": numerical,
            "Communication": communication,
            "Creativity": creativity,
            "Leadership": leadership
        }

        strong = []
        weak = []

        for skill, value in scores.items():

            if value >= 70:
                strong.append(skill)
            else:
                weak.append(skill)

        st.subheader("💪 Strong Skills")

        if strong:
            for s in strong:
                st.success(s)

        st.subheader("📈 Skills To Improve")

        if weak:
            for s in weak:
                st.warning(s)

        # -----------------------------
        # Performance Chart
        # -----------------------------
        st.subheader("📊 Performance Chart")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(scores.keys(), scores.values())
        ax.set_ylabel("Score")
        ax.set_title("Skill Scores")

        st.pyplot(fig)

        # -----------------------------
        # Career Tips
        # -----------------------------
        st.subheader("🚀 Career Tips")

        st.write("✅ Improve your skills regularly")
        st.write("✅ Build projects and portfolio")
        st.write("✅ Learn communication skills")
        st.write("✅ Keep exploring new technologies")

        st.balloons()

    except Exception as e:
        st.error(f"Prediction Error: {e}")

    
            
