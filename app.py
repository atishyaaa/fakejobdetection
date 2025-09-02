import streamlit as st
import pandas as pd
from predict import predict_fake_job

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="Honest Hire", layout="centered")

# ---------------------- Title and Subtitle ----------------------
st.title("Honest Hire")
st.markdown("""
Empowering recruiters and job seekers with accurate fraud detection.  
Upload job details individually or through a CSV file.
""")

# ---------------------- Single Job Prediction ----------------------
st.header("Verify a Job Posting")

title = st.text_input("Job Title")
company_profile = st.text_area("About the Company (optional)")
description = st.text_area("Job Description", height=120)
requirements = st.text_area("Job Requirements (optional)")

if st.button("Predict"):
    if not title.strip() or not description.strip():
        st.warning("Please provide at least the job title and description.")
    else:
        result, confidence = predict_fake_job(title, company_profile, description, requirements)
        st.success(f"Prediction: **{result}** ({confidence}% confidence)")
        if result == "Fake":
            st.error("❌ Warning: This job posting seems suspicious.")
        else:
            st.info("✅ This job posting seems genuine.")

# ---------------------- Bulk Upload ----------------------
st.markdown("---")
st.header("📁 Bulk CSV Prediction")
st.markdown("Upload a CSV with `title` and `description`. Optionally include `company_profile` and `requirements`.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'title' not in df.columns or 'description' not in df.columns:
            st.error("The CSV must contain at least 'title' and 'description' columns.")
        else:
            # Ensure optional fields are present
            if 'company_profile' not in df.columns:
                df['company_profile'] = ""
            if 'requirements' not in df.columns:
                df['requirements'] = ""

            results = []
            for _, row in df.iterrows():
                pred, conf = predict_fake_job(
                    str(row['title']),
                    str(row['company_profile']),
                    str(row['description']),
                    str(row['requirements'])
                )
                results.append({
                    "Title": row['title'],
                    "Description": row['description'],
                    "Prediction": pred,
                    "Confidence (%)": conf
                })

            result_df = pd.DataFrame(results)
            st.success("✅ Bulk predictions completed!")
            st.dataframe(result_df)

            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results", data=csv, file_name="fake_job_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
