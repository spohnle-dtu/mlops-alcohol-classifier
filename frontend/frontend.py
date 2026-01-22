import os
import requests
import streamlit as st

# Backend base URL, e.g. https://alcohol-api-xxxxx-ew.a.run.app
BACKEND = os.getenv("BACKEND", "").rstrip("/")

st.set_page_config(page_title="Alcohol Classifier", page_icon="🍺", layout="centered")
st.title("Alcohol Classifier")
st.write("Upload an image, and the model will predict **beer / wine / whiskey** (or your classes).")

if not BACKEND:
    st.error("BACKEND env var is not set. Set BACKEND to your FastAPI service URL.")
    st.stop()

st.caption(f"Backend: {BACKEND}")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image_bytes = uploaded.getvalue()
    st.image(image_bytes, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        with st.spinner("Running inference..."):
            try:
                # IMPORTANT: your FastAPI expects the field name "image"
                files = {"image": (uploaded.name, image_bytes, uploaded.type)}
                r = requests.post(f"{BACKEND}/predict", files=files, timeout=30)
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                st.stop()

        if r.status_code != 200:
            st.error(f"Backend returned {r.status_code}: {r.text}")
            st.stop()

        result = r.json()
        pred = result.get("predicted_class", "N/A")
        probs = result.get("probabilities", {})

        st.success(f"Prediction: **{pred}**")

        if isinstance(probs, dict) and probs:
            # Show as a nice bar chart
            st.subheader("Probabilities")
            st.bar_chart(probs)
        else:
            st.warning("No probabilities returned.")
            st.json(result)

