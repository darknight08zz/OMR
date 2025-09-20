import streamlit as st
import os
import shutil
import json
import pandas as pd
import cv2

# Import from main.py
from main import (
    preprocess_image,
    extract_answers,
    detect_version,
    load_exam_config,
    calculate_score,
    init_db,
    save_result_to_db
)

# ========================
# PAGE CONFIG & THEME
# ========================
st.set_page_config(
    page_title="🎓 OMR Auto Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        margin: 5px 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# SIDEBAR NAVIGATION
# ========================
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Innomatics+Logo", use_container_width=True)
    st.title("🎓 OMR Evaluator")
    st.markdown("---")
    st.markdown("### 🧭 Navigation")

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "upload"

    # Navigation buttons
    if st.button("📤 Upload Sheets", use_container_width=True,
                 type="primary" if st.session_state.current_page == "upload" else "secondary"):
        st.session_state.current_page = "upload"

    if st.button("📊 View Results", use_container_width=True,
                 type="primary" if st.session_state.current_page == "results" else "secondary"):
        st.session_state.current_page = "results"

    if st.button("📈 Analytics", use_container_width=True,
                 type="primary" if st.session_state.current_page == "analytics" else "secondary"):
        st.session_state.current_page = "analytics"

    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    st.caption("Version 2.0")
    st.caption("© 2025 Innomatics Research Labs")

# ========================
# PAGE FUNCTIONS
# ========================

def show_upload_page():
    st.title("📄 Automated OMR Evaluation System")
    st.markdown("### ⚡ Instantly grade OMR sheets with AI-powered accuracy")
    st.caption("Upload scanned OMR sheets → Get scores in seconds → Export reports → Track performance")

    st.markdown("---")
    st.markdown("## 📤 Upload OMR Sheets")
    uploaded_files = st.file_uploader(
        "Drag and drop OMR answer sheets here (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supports bulk upload. Max 20 files at once."
    )

    if uploaded_files:
        # Initialize DB
        init_db()
        results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {idx + 1} of {len(uploaded_files)}: {uploaded_file.name}")

            file_path = os.path.join("input", uploaded_file.name)
            
            # Ensure input directory exists
            os.makedirs("input", exist_ok=True)
            
            # Save uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with st.spinner(f"🧠 Analyzing {uploaded_file.name}..."):
                    # 1. Preprocess
                    processed_img, rectified_img = preprocess_image(file_path)

                    # 2. Extract answers
                    answers = extract_answers(processed_img, rectified_img)

                    # 3. Detect version
                    version = detect_version(processed_img)

                    # 4. Load exam config
                    exam_config_path = f"exams/math_exam_{version.lower()}.json"
                    if not os.path.exists(exam_config_path):
                        raise FileNotFoundError(f"Exam config not found: {exam_config_path}")

                    exam_config = load_exam_config(exam_config_path)

                    # 5. Calculate score
                    subject_scores, total_score, flagged = calculate_score(answers, exam_config)

                    # 6. Save debug image
                    os.makedirs("output", exist_ok=True)
                    debug_path = os.path.join("output", f"debug_{uploaded_file.name}")
                    if os.path.exists("output/debug_bubbles.jpg"):
                        debug_img = cv2.imread("output/debug_bubbles.jpg")
                        if debug_img is not None:
                            debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(debug_path, debug_img)

                    # 7. Prepare result
                    result = {
                        "StudentID": "UNKNOWN",
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
                    results.append(result)

                    # Show success
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ <strong>{uploaded_file.name}</strong> processed successfully!<br>
                    🎯 Score: <strong>{total_score}/100</strong> | Version: {version}
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                ❌ Error processing <strong>{uploaded_file.name}</strong>:<br>
                {str(e)}
                </div>
                """, unsafe_allow_html=True)
                continue

        # Store results for other pages
        st.session_state.processed_results = results
        st.success("🎉 All files processed successfully!")

    else:
        # Welcome screen
        st.markdown("## 👋 Welcome to OMR Auto Evaluator")
        


def show_results_page():
    st.title("📊 OMR Evaluation Results")

    if 'processed_results' not in st.session_state or not st.session_state.processed_results:
        st.info("📤 No results yet. Please upload and process OMR sheets first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = "upload"
            st.experimental_rerun()
        return

    results = st.session_state.processed_results
    df = pd.DataFrame(results)
    
    # Reorder columns
    if not df.empty:
        df_display = df[[
            "StudentID", "Filename", "Exam Name", "Version", 
            "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Total", "Flagged"
        ]]

        st.dataframe(
            df_display.style.format({
                "Subject1": "{:.0f}",
                "Subject2": "{:.0f}",
                "Subject3": "{:.0f}",
                "Subject4": "{:.0f}",
                "Subject5": "{:.0f}",
                "Total": "{:.0f}"
            }).applymap(
                lambda x: 'background-color: #d4edda' if isinstance(x, int) and x >= 15 else 
                         'background-color: #fff3cd' if isinstance(x, int) and x >= 10 else
                         'background-color: #f8d7da' if isinstance(x, int) else '',
                subset=["Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Total"]
            ),
            use_container_width=True,
            height=400
        )

        # Export button
        st.markdown("### 📥 Export Results")
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV Report",
            csv,
            "omr_results.csv",
            "text/csv",
            use_container_width=True
        )

        # Show debug images
        st.markdown("## 🔍 Bubble Detection Preview")
        st.caption("Green = Correctly detected marked bubble | Red = Unmarked")

        image_files = [res for res in results if res["Debug_Image"] and os.path.exists(res["Debug_Image"])]
        
        if image_files:
            cols = st.columns(2)
            for idx, res in enumerate(image_files):
                with cols[idx % 2]:
                    st.image(
                        res["Debug_Image"],
                        caption=f"{res['Filename']} - {res['Total']}/100",
                        use_container_width=True
                    )
        else:
            st.info("No debug images available.")


def show_analytics_page():
    st.title("📈 Performance Analytics Dashboard")

    if 'processed_results' not in st.session_state or not st.session_state.processed_results:
        st.info("📤 No data to analyze. Please upload and process OMR sheets first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = "upload"
            st.experimental_rerun()
        return

    df = pd.DataFrame(st.session_state.processed_results)
    if df.empty:
        st.warning("No data available for analytics.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Score Distribution")
        if not df["Total"].empty:
            st.bar_chart(df.set_index("Filename")["Total"], height=300)

    with col2:
        st.markdown("### 📚 Subject-wise Average")
        subject_cols = ["Subject1", "Subject2", "Subject3", "Subject4", "Subject5"]
        if all(col in df.columns for col in subject_cols):
            subject_avg = df[subject_cols].mean()
            st.bar_chart(subject_avg, height=300)

    # Performance bands
    st.markdown("### 🏅 Performance Categories")
    if "Total" in df.columns:
        df['Performance'] = pd.cut(
            df['Total'], 
            bins=[0, 40, 60, 80, 100], 
            labels=['🔴 Needs Improvement', '🟡 Average', '🟢 Good', '⭐ Excellent'],
            include_lowest=True
        )
        
        performance_counts = df['Performance'].value_counts()
        st.bar_chart(performance_counts, height=300)

    # Top performers
    st.markdown("### 🥇 Top Performers")
    if "Total" in df.columns:
        top_performers = df.nlargest(5, 'Total')[['StudentID', 'Filename', 'Total']]
        if not top_performers.empty:
            st.dataframe(top_performers, use_container_width=True, height=200)

# ========================
# ROUTER
# ========================
if st.session_state.current_page == "upload":
    show_upload_page()
elif st.session_state.current_page == "results":
    show_results_page()
elif st.session_state.current_page == "analytics":
    show_analytics_page()