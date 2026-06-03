import streamlit as st
import pandas as pd
import pickle
from career_info import career_info
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Career Prediction System",
    page_icon="🎯",
    layout="centered"
)

st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>
🎯 Career Prediction System 🚀
</h1>
""", unsafe_allow_html=True)

# Load Model
try:
    with open("career_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# Input Sliders
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

        st.markdown("---")

        st.markdown("""
        <h2 style='text-align:center; color:#ff6600;'>
        🏆 Top 3 Career Recommendations
        </h2>
        """, unsafe_allow_html=True)

        emojis = {
            "Data Analyst": "📊",
            "Graphic Designer": "🎨",
            "Software Developer": "💻",
            "Doctor": "🩺",
            "Teacher": "📚",
            "Engineer": "⚙️",
            "Lawyer": "⚖️",
            "Business Analyst": "💼",
            "Entrepreneur": "🚀",
            "Nurse": "🏥",
            "Scientist": "🔬"
        }

        for i, (career, prob) in enumerate(results, start=1):

            score = prob * 100
            emoji = emojis.get(career, "⭐")

            st.markdown(
                f"""
                <div style="
                    background:#f8f9fa;
                    padding:15px;
                    border-radius:15px;
                    margin-bottom:10px;
                    border-left:8px solid #4CAF50;
                ">
                    <h3>{i}. {emoji} {career}</h3>
                    <h4>⭐ Match Score: {score:.2f}%</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(prob))

        best_career = results[0][0]
        best_score = results[0][1] * 100

        st.success(
            f"🏆 Best Career Match: {best_career} ({best_score:.2f}%)"
        )

        st.info(
            "💡 This recommendation is based on your psychometric assessment."
        )

        st.markdown("""
        ### 🚀 Career Tips
        ✅ Improve your skills regularly  
        ✅ Build projects and portfolio  
        ✅ Learn communication skills  
        ✅ Keep exploring new technologies  
        """)

        st.balloons()
                # ===========================
        # Career Details
        # ===========================

        if best_career in career_info:

            st.subheader("📚 Recommended Skills")

            for skill in career_info[best_career]["skills"]:
                st.write(f"✅ {skill}")

            st.subheader("🛣️ Learning Roadmap")

            for step in career_info[best_career]["roadmap"]:
                st.write(f"➡️ {step}")

        # ===========================
        # Skill Analysis
        # ===========================

        scores = {
            "Logical": logical,
            "Numerical": numerical,
            "Communication": communication,
            "Creativity": creativity,
            "Leadership": leadership
        }

        strong = []
        weak = []

        for skill, score in scores.items():

            if score >= 70:
                strong.append(skill)
            else:
                weak.append(skill)

        st.subheader("💪 Strong Skills")

        for s in strong:
            st.success(s)

        st.subheader("📈 Skills To Improve")

        for s in weak:
            st.warning(s)

        # ===========================
        # Performance Chart
        # ===========================

        st.subheader("📊 Performance Chart")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            scores.keys(),
            scores.values()
        )

        ax.set_ylabel("Score")
        ax.set_title("Skill Scores")

        st.pyplot(fig)

        # ===========================
        # Confidence Meter
        # ===========================

        confidence = results[0][1] * 100

        st.metric(
            "🎯 Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # ===========================
        # Motivation Section
        # ===========================

        st.success(
            f"🚀 You are highly suited for a career in {best_career}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")