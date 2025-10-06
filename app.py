import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("simple_stress_model.pkl")
features = joblib.load("simple_model_features.pkl")

# Friendly labels for frontend
friendly_labels = {
    'sleep_duration': "💤 Sleep Duration (hours)",
    'screen_on_time': "📱 Daily Screen Time (hours)",
    'Neuroticism': "😟 Emotional Stability (Neuroticism, 1-5)",
    'Conscientiousness': "✅ Self-Discipline (Conscientiousness, 1-5)",
    'skin_conductance': "💧 Skin Conductance (sweating level, 0–5)",
    'accelerometer': "🏃‍♂️ Activity Level (movement intensity, 0–5)",
    'PSQI_score': "😴 Sleep Quality Score (1=Good, 4=Poor)"
}

# Streamlit App UI
st.set_page_config(page_title="Stress Level Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Smart Stress Level Predictor")
st.markdown("Enter your daily lifestyle & emotional parameters below:")

# Collect user inputs
user_input = {}
for feature in features:
    label = friendly_labels.get(feature, feature)
    # Use different input widgets depending on feature type
    if "hours" in label.lower():
        user_input[feature] = st.slider(label, 0.0, 12.0, 6.0)
    elif "1-5" in label:
        user_input[feature] = st.slider(label, 1.0, 5.0, 3.0)
    elif "score" in label.lower():
        user_input[feature] = st.slider(label, 1, 4, 2)
    else:
        user_input[feature] = st.number_input(label, value=0.0)

# Predict button
if st.button("🔍 Predict My Stress Level"):
    df = pd.DataFrame([user_input])
    df = df.reindex(columns=features, fill_value=0)
    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]
    conf = round(max(probs) * 100, 2)

    # Color-coded display for 5 levels
    color_labels = {
        "Very Low": "🟢 Very Low (You seem completely relaxed!)",
        "Low": "🟢 Low (You're doing great — calm and balanced!)",
        "Moderate": "🟡 Moderate (Take a short break today!)",
        "High": "🟠 High (You may need some rest or self-care.)",
        "Very High": "🔴 Very High (Please relax and take it easy!)"
    }

    st.subheader("Your Stress Level:")
    st.success(color_labels.get(pred, pred))
    st.write(f"**Model Confidence:** {conf}%")

    # Optional probability visualization
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(model.classes_, probs, color=['green', 'limegreen', 'gold', 'orange', 'red'])
    ax.set_ylabel("Probability")
    ax.set_xlabel("Stress Level")
    ax.set_title("Predicted Stress Probability Distribution")
    st.pyplot(fig)

    # Suggestion message
    if pred == "Very High":
        st.error("⚠️ High stress detected. Take time to rest, breathe, or meditate.")
    elif pred == "High":
        st.warning("💡 Tip: Go for a walk, listen to music, or take a short break.")
    elif pred == "Moderate":
        st.info("💡 Tip: You're doing okay, but balance work and rest.")
    elif pred == "Low":
        st.success("🌿 You’re calm and composed. Keep maintaining balance!")
    else:
        st.balloons()
        st.success("🎉 Totally relaxed! Great job taking care of yourself.")
