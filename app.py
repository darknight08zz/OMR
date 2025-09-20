import streamlit as st
import os
import shutil
import json
from main import (
    preprocess_image,
    extract_answers,
    detect_version,
    load_exam_config,
    calculate_score,
    init_db,
    save_result_to_db
)
import cv2
import pandas as pd

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="OMR Auto Evaluator", layout="centered")
st.title("📄 OMR Auto Evaluator (Production Ready)")

# ========================
# UPLOAD INTERFACE
# ========================
uploaded_files = st.file_uploader(
    "Upload OMR Answer Sheets",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ========================
# PROCESSING
# ========================
if uploaded_files:
    # Initialize DB once
    init_db()

    results = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join("input", uploaded_file.name)
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # 1. Preprocess image
                processed_img, rectified_img = preprocess_image(file_path)

                # 2. Extract answers
                answers = extract_answers(processed_img, rectified_img)

                # 3. Detect version (SET_A/B/C/D)
                version = detect_version(processed_img)

                # 4. Load exam config
                exam_config_path = f"exams/math_exam_{version.lower()}.json"
                if not os.path.exists(exam_config_path):
                    raise FileNotFoundError(f"Exam config not found: {exam_config_path}")

                exam_config = load_exam_config(exam_config_path)

                # 5. Calculate score
                subject_scores, total_score, flagged = calculate_score(answers, exam_config)

                # 6. Save debug image
                debug_path = os.path.join("output", f"debug_{uploaded_file.name}")
                if os.path.exists("output/debug_bubbles.jpg"):
                    # Convert to RGB for Streamlit
                    debug_img = cv2.imread("output/debug_bubbles.jpg")
                    debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(debug_path, debug_img)

                # 7. Prepare result
                result = {
                    "Filename": uploaded_file.name,
                    "Exam Name": exam_config["exam_name"],
                    "Exam ID": exam_config["exam_id"],
                    "Version": version,
                    "Subject1": subject_scores[0],
                    "Subject2": subject_scores[1],
                    "Subject3": subject_scores[2],
                    "Subject4": subject_scores[3],
                    "Subject5": subject_scores[4],
                    "Total": total_score,
                    "Flagged": flagged,
                    "Debug_Image": debug_path if os.path.exists(debug_path) else ""
                }

                # 8. Save to DB
                save_result_to_db(result)

                # 9. Add to results
                results.append(result)

                st.success(f"✅ {uploaded_file.name} → Score: {total_score}/100 (Version: {version})")

        except Exception as e:
            st.error(f"❌ {uploaded_file.name}: {str(e)}")
            continue

    # ========================
    # RESULTS DISPLAY
    # ========================
    if results:
        st.write("## 📊 Results Summary")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Display table
        st.dataframe(df.style.format({
            "Subject1": "{:.0f}",
            "Subject2": "{:.0f}",
            "Subject3": "{:.0f}",
            "Subject4": "{:.0f}",
            "Subject5": "{:.0f}",
            "Total": "{:.0f}"
        }))

        # Export CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Results (CSV)",
            csv,
            "omr_results.csv",
            "text/csv",
            key='download-csv'
        )

        # Show debug images
        st.write("## 🔍 Bubble Detection Preview")
        cols = st.columns(2)
        col_idx = 0

        for res in results:
            if res["Debug_Image"] and os.path.exists(res["Debug_Image"]):
                with cols[col_idx % 2]:
                    st.image(res["Debug_Image"], caption=f"{res['Filename']} ({res['Total']}/100)", use_column_width=True)
                col_idx += 1

else:
    st.info("👆 Upload one or more OMR images to begin.")
    