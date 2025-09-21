import streamlit as st
import os
import shutil
import json
import pandas as pd
import cv2
import numpy as np
from datetime import datetime

# JSON serialization helper function
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    else:
        return obj

# Import from the enhanced main.py with auto-crop features and answer set selection
# Note: Make sure main.py is in the same directory as this Streamlit app
try:
    from main import (
        create_answer_sets_input,
        load_or_create_answer_sets,
        select_answer_set_for_processing,
        calculate_score_with_comparison,
        process_single_image_with_autocrop,  # Updated function with answer set parameter
        batch_process_with_autocrop_and_selection,  # Updated batch function
        convert_to_greyscale,
        apply_enhancement_techniques,
        detect_and_crop_bubble_region,
        detect_bubbles_enhanced_crop,
        organize_bubbles_into_grid_enhanced,
        map_bubbles_to_questions_enhanced,
        detect_markings_enhanced_crop
    )
except ImportError as e:
    st.error(f"Error importing main.py functions: {e}")
    st.error("Make sure main.py is in the same directory as this Streamlit app")

# ========================
# PAGE CONFIG & THEME
# ========================
st.set_page_config(
    page_title="🎯 Automated OMR Evaluation & Scoring System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS with answer set selection theme
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
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .answer-set-selector {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .selected-set-display {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .autocrop-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .roi-detected {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .answer-set-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .comparison-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .pipeline-step {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-left: 4px solid #007bff;
        background-color: #f8f9fa;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .pipeline-step:hover {
        background-color: #e9ecef;
        border-left-color: #0056b3;
    }
    .crop-stats {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .set-a { border-left: 4px solid #28a745; background-color: #e8f5e8; }
    .set-b { border-left: 4px solid #007bff; background-color: #e3f2fd; }
    .temp-storage { border-left: 4px solid #ffc107; background-color: #fff8e1; }
    </style>
""", unsafe_allow_html=True)

# ========================
# SIDEBAR NAVIGATION
# ========================
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Auto-Crop+OMR+v3.0", use_container_width=True)
    st.title("Auto-Crop OMR with Answer Set Selection")
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "answer_sets"
    # Initialize answer set selection in session state
    if 'selected_answer_set' not in st.session_state:
        st.session_state.selected_answer_set = None
    # Enhanced navigation with answer set selection features
    pages = [
        ("answer_sets", "📝 Answer Sets", "Create & manage answer sets"),
        ("upload", "🎯 Image Processing", "Auto-crop with answer set selection"),
        ("results", "📊 View Results", "Results with answer set info"),
        ("comparison", "🔍 Answer Comparison", "Detailed comparison analysis"),
        ("debug", "🔍 Debug Pipeline", "Visual debugging"),
        ("analytics", "📈 Analytics", "Performance insights")
    ]
    # Navigation button logic with explicit rerun handling
    for page_key, button_text, description in pages:
        is_current = st.session_state.current_page == page_key
        button_key = f"nav_{page_key}"  # Unique key for each button
        if st.button(button_text, key=button_key, use_container_width=True,
                     type="primary" if is_current else "secondary"):
            st.session_state.current_page = page_key
            st.rerun()
        st.caption(description)
    st.markdown("---")
    st.markdown("### 🎯 Answer Set Selection")
    # Display current selected answer set
    if st.session_state.selected_answer_set:
        st.success(f"✅ Selected: {st.session_state.selected_answer_set}")
    else:
        st.info("📝 No answer set selected")
    st.markdown("### ⚙️ Auto-Crop Features")
    st.caption("🎯 Automatic Bubble Region Detection")
    st.caption("📏 Smart Image Cropping")
    st.caption("🔍 Enhanced Precision on ROI")
    st.caption("📊 Answer Set Selection")
    st.caption("⚡ Faster Processing")
    # Answer sets status
    if os.path.exists("answer_keys.json"):
        try:
            with open("answer_keys.json", "r") as f:
                sets = json.load(f)
            st.markdown("### 📋 Available Sets")
            for set_name in sets.keys():
                st.success(f"✅ {set_name}")
        except:
            st.error("⚠️ Sets Need Setup")

# ========================
# HELPER FUNCTIONS FOR ANSWER SET SELECTION
# ========================
def get_available_answer_sets():
    """Get available answer sets from file"""
    if os.path.exists("answer_keys.json"):
        try:
            with open("answer_keys.json", "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def display_answer_set_selector():
    """Display answer set selection interface"""
    answer_sets = get_available_answer_sets()
    if not answer_sets:
        st.warning("⚠️ No answer sets available. Please create answer sets first.")
        return None
    
    st.markdown("### 📝 Select Answer Set for Processing")
    # Create selection options
    set_options = list(answer_sets.keys())
    
    # Display current selection
    if st.session_state.selected_answer_set in set_options:
        default_index = set_options.index(st.session_state.selected_answer_set)
    else:
        default_index = 0
        
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_set = st.selectbox(
            "Choose answer set:",
            set_options,
            index=default_index,
            help="Select which answer set to use for scoring the processed images"
        )
    with col2:
        if st.button("🔄 Update Selection", use_container_width=True):
            st.session_state.selected_answer_set = selected_set
            st.success(f"Selected: {selected_set}")
            st.rerun()
    
    # Display preview of selected set
    if selected_set:
        st.markdown("### 📋 Answer Set Preview")
        preview_answers = answer_sets[selected_set]
        preview_text = ''.join(preview_answers[:20]) + "..." if len(preview_answers) > 20 else ''.join(preview_answers)
        st.markdown(f"""
        <div class="selected-set-display">
        <h4>📋 Selected: {selected_set}</h4>
        <p><strong>Preview (first 20):</strong> {preview_text}</p>
        <p><strong>Total Questions:</strong> {len(preview_answers)}</p>
        <p><strong>Answer Distribution:</strong> A: {preview_answers.count('A')}, B: {preview_answers.count('B')}, C: {preview_answers.count('C')}, D: {preview_answers.count('D')}</p>
        </div>
        """, unsafe_allow_html=True)
        
    return selected_set

# ========================
# PAGE FUNCTIONS
# ========================
def extract_answers_from_column(column):
    """Extract answers from a pandas column, handling various formats"""
    try:
        # Convert to list and clean
        answers = []
        for value in column.dropna():
            value_str = str(value).strip().upper()
            # Handle different formats
            if len(value_str) == 1 and value_str in ['A', 'B', 'C', 'D']:
                # Single answer per cell
                answers.append(value_str)
            elif len(value_str) > 1:
                # Multiple answers in one cell (like "ABCD" or "A,B,C,D")
                if ',' in value_str:
                    # Comma-separated
                    cell_answers = [ans.strip() for ans in value_str.split(',')]
                else:
                    # Continuous string
                    cell_answers = list(value_str.replace(' ', ''))
                # Add valid answers
                for ans in cell_answers:
                    if ans in ['A', 'B', 'C', 'D']:
                        answers.append(ans)
        # Validate total count and content
        if len(answers) == 100 and all(ans in ['A', 'B', 'C', 'D'] for ans in answers):
            return answers
        return None
    except Exception as e:
        print(f"Error extracting from column: {e}")
        return None

def extract_answers_from_dataframe(df):
    """Extract answers from a dataframe, trying different columns"""
    try:
        # First, try to find a column with answers
        for col in df.columns:
            answers = extract_answers_from_column(df[col])
            if answers:
                return answers
        # If no single column works, try concatenating first row
        if len(df) > 0:
            first_row_answers = []
            for col in df.columns:
                value = df[col].iloc[0]
                if pd.notna(value):
                    value_str = str(value).strip().upper()
                    if value_str in ['A', 'B', 'C', 'D']:
                        first_row_answers.append(value_str)
            if len(first_row_answers) == 100:
                return first_row_answers
        return None
    except Exception as e:
        print(f"Error extracting from dataframe: {e}")
        return None

def show_answer_sets_page():
    st.title("📝 Answer Sets Management")
    st.markdown("### Create and manage your answer sets for auto-crop OMR processing")
    
    # Answer sets overview with answer set selection mention
    with st.container():
        st.markdown("""
        <div class="answer-set-box">
        <h4>📋 Enhanced Answer Set System with Selection</h4>
        <p><strong>Multiple Sets:</strong> Create multiple answer sets (A, B, C, etc.)</p>
        <p><strong>Selection Interface:</strong> Choose which answer set to use for processing</p>
        <p><strong>Auto-Crop Ready:</strong> Optimized for automatic bubble region detection</p>
        <p><strong>Enhanced Processing:</strong> Works with cropped bubble regions for better accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check current answer sets
    current_sets = {}
    if os.path.exists("answer_keys.json"):
        try:
            with open("answer_keys.json", "r") as f:
                current_sets = json.load(f)
        except Exception as e:
            st.error(f"Error reading answer sets: {e}")
    
    # Display current sets status
    if current_sets:
        st.markdown("### 📋 Current Answer Sets")
        cols = st.columns(min(len(current_sets), 4))
        for i, (set_name, answers) in enumerate(current_sets.items()):
            with cols[i % 4]:
                st.markdown(f'<div class="success-box set-a">✅ {set_name} configured</div>', unsafe_allow_html=True)
                preview = ''.join(answers[:10]) + "..." if len(answers) > 10 else ''.join(answers)
                st.code(f"Preview: {preview}", language=None)
                # Answer distribution
                dist = {opt: answers.count(opt) for opt in ['A', 'B', 'C', 'D']}
                st.caption(f"A:{dist['A']} B:{dist['B']} C:{dist['C']} D:{dist['D']}")
    else:
        st.markdown('<div class="error-box">❌ No answer sets configured</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Answer set creation interface
    st.markdown("### 📝 Create/Update Answer Sets")
    tab1, tab2 = st.tabs(["📋 Interactive Input", "📁 File Upload"])
    
    with tab1:
        st.markdown("#### Enter Answer Sets Manually")
        with st.form("answer_sets_form"):
            st.info("Enter 100 answers (A, B, C, or D) for each set. You can create multiple sets.")
            # Dynamic answer set creation
            num_sets = st.number_input("Number of answer sets to create:", min_value=1, max_value=10, value=2)
            new_sets = {}
            for i in range(num_sets):
                set_name = st.text_input(f"Set {i+1} Name:", value=f"SET_{chr(65+i)}", key=f"set_name_{i}")
                set_answers = st.text_area(
                    f"{set_name} Answers (100 answers):",
                    placeholder="ABCDABCD... or A,B,C,D,A,B,C,D...",
                    height=80,
                    help="Enter exactly 100 answers using A, B, C, or D",
                    key=f"set_answers_{i}"
                )
                if set_answers.strip():
                    # Process answers
                    if ',' in set_answers:
                        answers = [ans.strip().upper() for ans in set_answers.split(',')]
                    else:
                        answers = list(set_answers.strip().upper().replace(' ', ''))
                    if len(answers) == 100 and all(ans in ['A', 'B', 'C', 'D'] for ans in answers):
                        new_sets[set_name] = answers
                    else:
                        st.error(f"{set_name}: Invalid format (need exactly 100 A/B/C/D answers)")
            submitted = st.form_submit_button("💾 Save Answer Sets", use_container_width=True)
            if submitted and new_sets:
                try:
                    # Merge with existing sets
                    if current_sets:
                        current_sets.update(new_sets)
                    else:
                        current_sets = new_sets
                    with open("answer_keys.json", "w") as f:
                        json.dump(current_sets, f, indent=2)
                    st.success(f"✅ Successfully saved {len(new_sets)} answer set(s)!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving answer sets: {e}")
    
    with tab2:
        st.markdown("#### Upload Answer Sets from File")
        st.info("Upload JSON, Excel (.xlsx), or CSV files with answer sets")
        uploaded_file = st.file_uploader(
            "Choose answer file", 
            type=["json", "xlsx", "csv"],
            help="Supports JSON, Excel with columns, or CSV with answer data"
        )
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            try:
                valid_sets = {}
                if file_type == "json":
                    content = json.load(uploaded_file)
                    for set_name, answers in content.items():
                        if isinstance(answers, list) and len(answers) == 100:
                            if all(ans in ['A', 'B', 'C', 'D'] for ans in answers):
                                valid_sets[set_name] = answers
                            else:
                                st.error(f"{set_name}: Invalid answers (must be A/B/C/D only)")
                        else:
                            st.error(f"{set_name}: Invalid format (need exactly 100 answers)")
                elif file_type in ["xlsx", "csv"]:
                    if file_type == "xlsx":
                        df = pd.read_excel(uploaded_file, sheet_name=None)
                        if len(df) > 1:
                            # Multiple sheets
                            for sheet_name, sheet_df in df.items():
                                answers = extract_answers_from_dataframe(sheet_df)
                                if answers and len(answers) == 100:
                                    clean_name = f"SET_{sheet_name.upper()}"
                                    valid_sets[clean_name] = answers
                                    st.success(f"✅ Extracted {clean_name} from sheet: {sheet_name}")
                        else:
                            # Single sheet with columns
                            sheet_df = list(df.values())[0]
                            for col in sheet_df.columns:
                                answers = extract_answers_from_column(sheet_df[col])
                                if answers and len(answers) == 100:
                                    clean_name = f"SET_{str(col).upper()}"
                                    valid_sets[clean_name] = answers
                                    st.success(f"✅ Extracted {clean_name} from column: {col}")
                    else:  # CSV
                        df = pd.read_csv(uploaded_file)
                        for col in df.columns:
                            answers = extract_answers_from_column(df[col])
                            if answers and len(answers) == 100:
                                clean_name = f"SET_{str(col).upper()}"
                                valid_sets[clean_name] = answers
                                st.success(f"✅ Extracted {clean_name} from column: {col}")
                
                if valid_sets:
                    st.markdown("### 📋 Preview of Extracted Answer Sets:")
                    for set_name, answers in valid_sets.items():
                        with st.expander(f"Preview {set_name} ({len(answers)} answers)"):
                            preview = ''.join(answers[:50]) + "..." if len(answers) > 50 else ''.join(answers)
                            st.code(preview, language=None)
                            answer_counts = pd.Series(answers).value_counts()
                            st.write("Answer distribution:", dict(answer_counts))
                    
                    if st.button("📥 Import Extracted Answer Sets"):
                        # Merge with existing
                        current_sets.update(valid_sets)
                        with open("answer_keys.json", "w") as f:
                            json.dump(current_sets, f, indent=2)
                        st.success(f"✅ Imported {len(valid_sets)} answer set(s)!")
                        st.rerun()
                else:
                    st.error("❌ Could not extract valid answer sets from the file.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Download and management section
    if current_sets:
        st.markdown("---")
        st.markdown("### 📥 Export and Manage Answer Sets")
        col1, col2, col3 = st.columns(3)
        with col1:
            # Export all sets
            json_safe_sets = convert_numpy_types(current_sets)
            json_data = json.dumps(json_safe_sets, indent=2)
            st.download_button(
                "⬇️ Download All Answer Sets",
                json_data,
                f"answer_sets_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json",
                use_container_width=True
            )
        with col2:
            # Delete sets
            if len(current_sets) > 0:
                set_to_delete = st.selectbox("Select set to delete:", list(current_sets.keys()))
                if st.button("🗑️ Delete Selected Set", use_container_width=True):
                    del current_sets[set_to_delete]
                    with open("answer_keys.json", "w") as f:
                        json.dump(current_sets, f, indent=2)
                    st.success(f"Deleted {set_to_delete}")
                    st.rerun()
        with col3:
            # Clear all sets
            if st.button("⚠️ Clear All Sets", use_container_width=True, type="secondary"):
                if os.path.exists("answer_keys.json"):
                    os.remove("answer_keys.json")
                st.success("All answer sets cleared")
                st.rerun()

def show_upload_page():
    st.title("🎯 OMR Processing with Answer Set Selection")
    st.markdown("### Auto-crop processing with manual answer set selection")
    
    # Check if answer sets exist
    answer_sets = get_available_answer_sets()
    if not answer_sets:
        st.error("❌ No answer sets found! Please create answer sets first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Go to Answer Sets", use_container_width=True):
                st.session_state.current_page = "answer_sets"
                st.rerun()
        with col2:
            if st.button("🚀 Create Default Sets", use_container_width=True):
                # Create default sets for demo
                default_sets = {
                    "SET_A": ["A", "B", "C", "D"] * 25,
                    "SET_B": ["D", "C", "B", "A"] * 25
                }
                with open("answer_keys.json", "w") as f:
                    json.dump(default_sets, f, indent=2)
                st.success("✅ Created default sets")
                st.rerun()
        return
    
    # Answer set selection interface
    st.markdown("### 📝 Step 1: Select Answer Set for Processing")
    selected_answer_set = display_answer_set_selector()
    if not selected_answer_set:
        return
    
    # Update session state with selection
    st.session_state.selected_answer_set = selected_answer_set
    
    st.markdown("---")
    
    # Auto-crop processing info
    st.markdown("""
    <div class="autocrop-box">
    <h4>🎯 Auto-Crop Enhanced Processing Pipeline</h4>
    <p><strong>Answer Set Selection:</strong> Manual selection of answer set for accurate scoring</p>
    <p><strong>Automatic Detection:</strong> Smart bubble region identification and cropping</p>
    <p><strong>Enhanced Precision:</strong> Optimized algorithms for cropped bubble regions</p>
    <p><strong>Flexible Scoring:</strong> Use any available answer set for evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎯 Step 2: Upload OMR Sheets")
    uploaded_files = st.file_uploader(
        "Upload OMR answer sheets (JPG/PNG/TIFF)",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=True,
        help=f"Images will be processed using answer set: {selected_answer_set}"
    )
    
    if uploaded_files:
        st.markdown(f"""
        <div class="answer-set-selector">
        <h4>📝 Processing Configuration</h4>
        <p><strong>Selected Answer Set:</strong> {selected_answer_set}</p>
        <p><strong>Number of Files:</strong> {len(uploaded_files)}</p>
        <p><strong>Processing Method:</strong> Auto-Crop Enhanced Pipeline</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Processing confirmation
        if st.button("🚀 Start Processing with Selected Answer Set", use_container_width=True, type="primary"):
            results = []
            temp_storage_log = []
            crop_statistics = []
            
            # Processing progress
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                pipeline_status = st.empty()
                crop_metrics = st.empty()
                
            for idx, uploaded_file in enumerate(uploaded_files):
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Save uploaded file
                os.makedirs("input", exist_ok=True)
                file_path = os.path.join("input", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    with st.spinner(f"🎯 Processing: {uploaded_file.name} with {selected_answer_set}..."):
                        temp_storage = []
                        # Process with selected answer set
                        result, detailed_results = process_single_image_with_autocrop(
                            file_path, selected_answer_set, answer_sets, temp_storage
                        )
                        if result:
                            result.update({
                                "StudentID": f"STUDENT_{str(idx + 1).zfill(3)}",
                                "Exam_Name": "AUTO_CROP_OMR",
                                "Processing_Method": f"Auto-Crop with {selected_answer_set}",
                                "Temp_Storage_Size": len(temp_storage)
                            })
                            results.append(result)
                            temp_storage_log.append({
                                'filename': uploaded_file.name,
                                'temp_storage': temp_storage.copy(),
                                'final_score': result['Total'],
                                'answer_set_used': selected_answer_set,
                                'auto_cropped': result.get('Auto_Cropped', False)
                            })
                            
                            # Track crop statistics
                            if result.get('Auto_Cropped', False):
                                crop_statistics.append({
                                    'filename': uploaded_file.name,
                                    'reduction_ratio': result.get('Crop_Reduction', 0),
                                    'bubble_density': result.get('Bubble_Density', 0),
                                    'grid_quality': result.get('Grid_Quality', 0),
                                    'answer_set_used': selected_answer_set
                                })
                            
                            # Real-time metrics display
                            with crop_metrics.container():
                                cols = st.columns(5)
                                with cols[0]:
                                    st.metric("Answer Set", selected_answer_set)
                                with cols[1]:
                                    st.metric("Score", f"{result['Total']}/100")
                                with cols[2]:
                                    crop_status = "🎯 CROPPED" if result.get('Auto_Cropped', False) else "📄 Full"
                                    st.metric("Processing", crop_status)
                                with cols[3]:
                                    reduction = result.get('Crop_Reduction', 0)
                                    st.metric("Size Reduction", f"{reduction:.1%}")
                                with cols[4]:
                                    grid_quality = result.get('Grid_Quality', 0)
                                    st.metric("Quality", f"{grid_quality:.3f}")
                            
                            # Success display
                            crop_indicator = "🎯 AUTO-CROPPED" if result.get('Auto_Cropped', False) else "📄 FULL IMAGE"
                            st.markdown(f"""
                            <div class="roi-detected">
                            ✅ <strong>{uploaded_file.name}</strong> - Processing Complete!<br>
                            📝 Answer Set: <strong>{selected_answer_set}</strong><br>
                            🎯 Score: <strong>{result['Total']}/100 ({result['Accuracy']:.1f}%)</strong><br>
                            📚 Subject Scores: {result['Subject1']}, {result['Subject2']}, {result['Subject3']}, {result['Subject4']}, {result['Subject5']}<br>
                            🎯 Processing: <strong>{crop_indicator}</strong><br>
                            📦 Temporary Storage: <strong>{len(temp_storage)} answers</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"❌ Processing failed for {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    continue
            
            # Processing completion
            pipeline_status.empty()
            crop_metrics.empty()
            status_text.empty()
            progress_bar.empty()
            
            if results:
                # Summary display
                avg_score = sum(r["Total"] for r in results) / len(results)
                cropped_files = len([r for r in results if r.get('Auto_Cropped', False)])
                st.markdown(f"""
                <div class="autocrop-box">
                <h3>🎉 Processing Complete with {selected_answer_set}!</h3>
                <p><strong>Files Processed:</strong> {len(results)}</p>
                <p><strong>Answer Set Used:</strong> {selected_answer_set}</p>
                <p><strong>Average Score:</strong> {avg_score:.1f}/100</p>
                <p><strong>Auto-Crop Success:</strong> {cropped_files}/{len(results)} files</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Store results
            st.session_state.processed_results = results
            st.session_state.temp_storage_log = temp_storage_log
            st.session_state.crop_statistics = crop_statistics
    else:
        # Welcome display
        st.markdown("## 👋 Ready for Processing!")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            ### 📝 Current Configuration
            - **Selected Answer Set:** {selected_answer_set}
            - **Processing Method:** Auto-Crop Enhanced
            - **Features:** Smart cropping + Enhanced precision
            - **Status:** Ready to process images
            """)
        with col2:
            st.markdown("""
            ### 🎯 Processing Steps
            1. **Upload Images** - Select your OMR sheets
            2. **Auto-Crop Detection** - Identify bubble regions
            3. **Enhanced Processing** - Optimized bubble detection
            4. **Answer Comparison** - Score against selected answer set
            5. **Results Generation** - Detailed scoring and metrics
            """)

def show_results_page():
    st.title("📊 Results with Answer Set Information")
    if 'processed_results' not in st.session_state or not st.session_state.processed_results:
        st.info("📤 No results available. Process OMR sheets first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = "upload"
            st.rerun()
        return
    
    results = st.session_state.processed_results
    df = pd.DataFrame(results)
    
    # Enhanced dashboard metrics
    st.markdown("### 🎯 Processing Results Dashboard")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📄 Total Sheets", len(results))
    with col2:
        avg_score = df["Total"].mean() if not df.empty else 0
        st.metric("📊 Average Score", f"{avg_score:.1f}/100")
    with col3:
        if 'Selected_Answer_Set' in df.columns:
            answer_set_used = df['Selected_Answer_Set'].iloc[0]
            st.metric("📝 Answer Set", answer_set_used)
    with col4:
        cropped_count = len(df[df.get('Auto_Cropped', pd.Series([False] * len(df)))])
        st.metric("🎯 Auto-Cropped", f"{cropped_count}/{len(results)}")
    with col5:
        flagged_count = df.get("Flagged", pd.Series(dtype=bool)).sum() if not df.empty else 0
        st.metric("⚠️ Flagged", flagged_count)
    
    # Results table with answer set information
    if not df.empty:
        display_columns = ["Filename", "Selected_Answer_Set", "Subject1", "Subject2", "Subject3", 
                          "Subject4", "Subject5", "Total", "Accuracy", "Auto_Cropped"]
        available_columns = [col for col in display_columns if col in df.columns]
        if available_columns:
            st.dataframe(df[available_columns], use_container_width=True, height=400)
        
        # Export with answer set info
        st.markdown("### 📥 Export Results")
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            f"omr_results_with_answersets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )

def show_comparison_page():
    """Enhanced comparison analysis page with answer set information"""
    st.title("🔍 Answer Comparison Analysis with Answer Set Details")
    st.markdown("### Detailed index-by-index comparison with answer set and crop performance data")
    
    if 'processed_results' not in st.session_state or not st.session_state.processed_results:
        st.info("📤 No comparison data available. Process OMR sheets first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = "upload"
            st.rerun()
        return
    
    # Load comparison log if available
    if os.path.exists("output/answer_comparison_log.csv"):
        comparison_df = pd.read_csv("output/answer_comparison_log.csv")
        # Add answer set information if available
        if 'processed_results' in st.session_state and st.session_state.processed_results:
            results_df = pd.DataFrame(st.session_state.processed_results)
            if 'Selected_Answer_Set' in results_df.columns:
                answer_set_used = results_df['Selected_Answer_Set'].iloc[0]
                comparison_df['answer_set_used'] = answer_set_used
        
        st.markdown("### 📊 Comparison Overview with Answer Set Information")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            total_comparisons = len(comparison_df)
            st.metric("Total Comparisons", total_comparisons)
        with col2:
            correct_count = len(comparison_df[comparison_df['is_correct']])
            st.metric("Correct Answers", f"{correct_count}/{total_comparisons}")
        with col3:
            flagged_count = len(comparison_df[comparison_df['is_flagged']])
            st.metric("Flagged Questions", flagged_count)
        with col4:
            accuracy = (correct_count / total_comparisons * 100) if total_comparisons > 0 else 0
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with col5:
            # Show answer set used
            if 'answer_set_used' in comparison_df.columns:
                answer_set = comparison_df['answer_set_used'].iloc[0] if not comparison_df.empty else "N/A"
                st.metric("Answer Set Used", answer_set)
            else:
                st.metric("Answer Set", "Not Available")
        
        # Answer set performance summary
        if 'answer_set_used' in comparison_df.columns:
            st.markdown("### 📝 Answer Set Performance Summary")
            answer_set = comparison_df['answer_set_used'].iloc[0]
            st.markdown(f"""
            <div class="selected-set-display">
            <h4>📝 Performance Analysis for {answer_set}</h4>
            <p><strong>Total Questions:</strong> {total_comparisons}</p>
            <p><strong>Correct Answers:</strong> {correct_count} ({accuracy:.1f}%)</p>
            <p><strong>Incorrect Answers:</strong> {total_comparisons - correct_count - flagged_count}</p>
            <p><strong>Flagged (Unanswered):</strong> {flagged_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Auto-crop performance summary (if available)
        if 'crop_statistics' in st.session_state and st.session_state.crop_statistics:
            st.markdown("### 🎯 Auto-Crop Performance with Answer Set")
            crop_stats = st.session_state.crop_statistics
            avg_reduction = sum(stat['reduction_ratio'] for stat in crop_stats) / len(crop_stats)
            avg_density = sum(stat['bubble_density'] for stat in crop_stats) / len(crop_stats)
            avg_quality = sum(stat['grid_quality'] for stat in crop_stats) / len(crop_stats)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="crop-stats">
                <strong>Auto-Crop Statistics:</strong><br>
                📏 Average Size Reduction: <strong>{avg_reduction:.1%}</strong><br>
                🎯 Average Bubble Density: <strong>{avg_density:.1f}/10k pixels</strong><br>
                📊 Average Grid Quality: <strong>{avg_quality:.3f}</strong><br>
                🔍 Files Successfully Cropped: <strong>{len(crop_stats)}</strong>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                # Show answer set correlation with crop performance
                if 'answer_set_used' in comparison_df.columns:
                    answer_set = comparison_df['answer_set_used'].iloc[0]
                    st.markdown(f"""
                    <div class="answer-set-selector">
                    <strong>Answer Set & Crop Correlation:</strong><br>
                    📝 Answer Set: <strong>{answer_set}</strong><br>
                    🎯 Crop Success Rate: <strong>{len(crop_stats)}/{len(st.session_state.processed_results)} files</strong><br>
                    📈 Accuracy with Cropping: <strong>{accuracy:.1f}%</strong><br>
                    ⚡ Enhanced Processing: <strong>Enabled</strong>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed comparison results with answer set info
        st.markdown("### 📋 Detailed Comparison Results")
        # Add comparison result column for better visualization
        def create_comparison_result(row):
            if row['is_correct']:
                return '✓ CORRECT'
            elif row['is_flagged']:
                return '? UNANSWERED'
            else:
                return '✗ INCORRECT'
        comparison_df['comparison_result'] = comparison_df.apply(create_comparison_result, axis=1)
        
        # Color coding function
        def color_comparison_result(val):
            if val == '✓ CORRECT':
                return 'background-color: #d4edda; color: #155724'
            elif val == '✗ INCORRECT':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == '? UNANSWERED':
                return 'background-color: #fff3cd; color: #856404'
            return ''
        
        # Filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_filter = st.selectbox(
                "Filter Results:",
                ["All", "Correct Only", "Incorrect Only", "Unanswered Only"]
            )
        with col2:
            subject_filter = st.selectbox(
                "Subject Filter:",
                ["All Subjects", "Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5"]
            )
        with col3:
            if 'answer_set_used' in comparison_df.columns:
                st.info(f"Answer Set: {comparison_df['answer_set_used'].iloc[0]}")
        
        # Apply filters
        filtered_df = comparison_df.copy()
        if show_filter != "All":
            if show_filter == "Correct Only":
                filtered_df = filtered_df[filtered_df['is_correct'] == True]
            elif show_filter == "Incorrect Only":
                filtered_df = filtered_df[(filtered_df['is_correct'] == False) & (filtered_df['is_flagged'] == False)]
            elif show_filter == "Unanswered Only":
                filtered_df = filtered_df[filtered_df['is_flagged'] == True]
        if subject_filter != "All Subjects":
            subject_num = int(subject_filter.split()[-1])
            filtered_df = filtered_df[filtered_df['subject'] == subject_num]
        
        # Display filtered results
        display_cols = ['question', 'subject', 'temp_answer', 'key_answer', 'comparison_result']
        if 'answer_set_used' in filtered_df.columns:
            display_cols.append('answer_set_used')
        if not filtered_df.empty:
            st.dataframe(
                filtered_df[display_cols].style.applymap(
                    color_comparison_result, subset=['comparison_result']
                ),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No results match the current filter criteria.")
        
        # Subject-wise analysis with answer set info
        st.markdown("### 📚 Subject-wise Analysis by Answer Set")
        subject_analysis = comparison_df.groupby('subject').agg({
            'is_correct': ['sum', 'count'],
            'is_flagged': 'sum'
        }).round(2)
        subject_analysis.columns = ['Correct', 'Total', 'Flagged']
        subject_analysis['Accuracy%'] = (subject_analysis['Correct'] / subject_analysis['Total'] * 100).round(1)
        subject_analysis['Subject'] = [f"Subject {i}" for i in range(1, 6)]
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(subject_analysis.set_index('Subject')['Accuracy%'])
        with col2:
            st.dataframe(subject_analysis, use_container_width=True)
        
        # Enhanced download section with answer set info
        st.markdown("### 📥 Export Comparison Data with Answer Set Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Filtered Results",
                csv_data,
                f"comparison_filtered_with_answerset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            full_csv = comparison_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Full Comparison",
                full_csv,
                f"full_comparison_with_answerset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col3:
            # Export answer set performance summary
            if 'answer_set_used' in comparison_df.columns:
                summary_data = {
                    'answer_set': comparison_df['answer_set_used'].iloc[0],
                    'total_questions': total_comparisons,
                    'correct_answers': correct_count,
                    'accuracy_percent': accuracy,
                    'flagged_questions': flagged_count,
                    'export_timestamp': datetime.now().isoformat()
                }
                summary_json = json.dumps(summary_data, indent=2)
                st.download_button(
                    "⬇️ Download Answer Set Summary",
                    summary_json,
                    f"answer_set_performance_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    "application/json",
                    use_container_width=True
                )
    else:
        st.warning("No comparison log found. Process some OMR sheets first to see detailed comparison analysis.")

def show_debug_page():
    st.title("🔍 Auto-Crop Enhanced Pipeline Debug Viewer")
    st.markdown("### Visual inspection of the auto-crop processing pipeline with answer set information")
    
    # Show answer set information if available
    if 'processed_results' in st.session_state and st.session_state.processed_results:
        results_df = pd.DataFrame(st.session_state.processed_results)
        if 'Selected_Answer_Set' in results_df.columns:
            answer_set_used = results_df['Selected_Answer_Set'].iloc[0]
            st.markdown(f"""
            <div class="answer-set-selector">
            <h4>📝 Debug Session Information</h4>
            <p><strong>Answer Set Used:</strong> {answer_set_used}</p>
            <p><strong>Processing Method:</strong> Auto-Crop Enhanced Pipeline</p>
            <p><strong>Files Processed:</strong> {len(results_df)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    if not os.path.exists("output/debug_steps"):
        st.info("📤 No debug images available. Process OMR sheets first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = "upload"
            st.rerun()
        return
    
    # Enhanced debug file mapping with auto-crop steps
    step_files = {
        "Step 1: Smart Greyscale": "step1_greyscale.jpg",
        "Step 2a: Advanced Denoising": "step2a_denoised.jpg", 
        "Step 2b: Multi-Enhancement": "step2b_enhanced.jpg",
        "Step 2c: Edge Preservation": "step2c_filtered.jpg",
        "Step 2d: Optimal Thresholding": "step2d_binary.jpg",
        "Step 2e: Morphological Cleanup": "step2e_cleaned.jpg",
        "🎯 Step 2.5a: Auto-Crop Detection": "step2_5_auto_crop_detection.jpg",
        "🎯 Step 2.5b: Cropped Binary": "step2_5_cropped_binary.jpg", 
        "🎯 Step 2.5c: Cropped Original": "step2_5_cropped_original.jpg",
        "Step 3: Enhanced Bubble Detection": "step3_enhanced_detection_cropped.jpg",
        "Step 6: Enhanced Marking": "step6_enhanced_marking_detection.jpg",
        "Alternative Processing": "step2_alternative.jpg"
    }
    
    available_steps = {}
    for step_name, filename in step_files.items():
        filepath = os.path.join("output/debug_steps", filename)
        if os.path.exists(filepath):
            available_steps[step_name] = filepath
    
    if not available_steps:
        st.warning("No debug step images found.")
        return
    
    # Enhanced step selector with auto-crop highlighting
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_step = st.selectbox(
            "Select auto-crop processing step:",
            list(available_steps.keys()),
            help="🎯 indicates auto-crop specific steps"
        )
    with col2:
        # Show processing statistics if available
        if 'crop_statistics' in st.session_state and st.session_state.crop_statistics:
            latest_stats = st.session_state.crop_statistics[-1]
            st.metric("Size Reduction", f"{latest_stats['reduction_ratio']:.1%}")
            st.metric("Grid Quality", f"{latest_stats['grid_quality']:.3f}")
    
    if selected_step:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(
                available_steps[selected_step],
                caption=f"Auto-crop processing: {selected_step}",
                use_container_width=True
            )
        with col2:
            st.markdown("### 📋 Step Details")
            step_descriptions = {
                "Step 1: Smart Greyscale": "Optimal greyscale conversion for auto-crop pipeline",
                "Step 2a: Advanced Denoising": "Enhanced denoising for better bubble detection",
                "Step 2b: Multi-Enhancement": "Multi-stage enhancement optimized for cropping",
                "Step 2c: Edge Preservation": "Edge preservation for accurate bubble boundaries", 
                "Step 2d: Optimal Thresholding": "Optimal thresholding for bubble region detection",
                "Step 2e: Morphological Cleanup": "Enhanced cleanup for improved bubble quality",
                "🎯 Step 2.5a: Auto-Crop Detection": "Automatic bubble region detection and boundary identification",
                "🎯 Step 2.5b: Cropped Binary": "Binary image after auto-crop extraction",
                "🎯 Step 2.5c: Cropped Original": "Original image after auto-crop extraction",
                "Step 3: Enhanced Bubble Detection": "Enhanced detection on cropped bubble region",
                "Step 6: Enhanced Marking": "Enhanced marking detection with auto-crop optimization",
                "Alternative Processing": "Fallback processing for challenging images"
            }
            description = step_descriptions.get(selected_step, "Auto-crop processing step")
            st.info(description)
            # Show auto-crop specific metrics if available
            if "🎯" in selected_step and 'crop_statistics' in st.session_state:
                st.markdown("**Auto-Crop Metrics:**")
                if st.session_state.crop_statistics:
                    latest_stats = st.session_state.crop_statistics[-1]
                    st.metric("Bubble Density", f"{latest_stats['bubble_density']:.1f}")
                    # Show answer set used for this processing
                    if 'answer_set_used' in latest_stats:
                        st.info(f"Answer Set: {latest_stats['answer_set_used']}")
    
    # Debug session summary
    st.markdown("### 📊 Debug Session Summary")
    if 'processed_results' in st.session_state and st.session_state.processed_results:
        results_df = pd.DataFrame(st.session_state.processed_results)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Processing Summary:**")
            cropped_count = len(results_df[results_df.get('Auto_Cropped', False)])
            st.write(f"- Files processed: {len(results_df)}")
            st.write(f"- Auto-cropped: {cropped_count}")
            st.write(f"- Average score: {results_df['Total'].mean():.1f}/100")
        with col2:
            st.markdown("**Answer Set Information:**")
            if 'Selected_Answer_Set' in results_df.columns:
                answer_set = results_df['Selected_Answer_Set'].iloc[0]
                st.write(f"- Answer set used: {answer_set}")
                st.write(f"- Processing method: Auto-crop enhanced")
                st.write(f"- Debug images: {len(available_steps)} steps")
        with col3:
            st.markdown("**Quality Metrics:**")
            if 'Grid_Quality' in results_df.columns:
                avg_quality = results_df['Grid_Quality'].mean()
                st.write(f"- Average grid quality: {avg_quality:.3f}")
            if 'crop_statistics' in st.session_state:
                crop_success_rate = len(st.session_state.crop_statistics) / len(results_df) * 100
                st.write(f"- Crop success rate: {crop_success_rate:.1f}%")

def show_analytics_page():
    st.title("📈 Analytics with Answer Set Performance Metrics")
    st.markdown("### Comprehensive performance analysis including answer set effectiveness")
    
    if 'processed_results' not in st.session_state or not st.session_state.processed_results:
        st.info("📤 No analytics data available. Process OMR sheets first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = "upload"
            st.rerun()
        return
    
    df = pd.DataFrame(st.session_state.processed_results)
    if df.empty:
        st.warning("No data available for analytics.")
        return
    
    # Answer Set Performance Analytics
    st.markdown("### 📝 Answer Set Performance Analysis")
    if 'Selected_Answer_Set' in df.columns:
        answer_set_used = df['Selected_Answer_Set'].iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="selected-set-display">
            <h4>📝 Answer Set: {answer_set_used}</h4>
            <p><strong>Files Processed:</strong> {len(df)}</p>
            <p><strong>Average Score:</strong> {df['Total'].mean():.1f}/100</p>
            <p><strong>Highest Score:</strong> {df['Total'].max()}/100</p>
            <p><strong>Lowest Score:</strong> {df['Total'].min()}/100</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            # Score distribution analysis
            st.markdown("**Score Distribution Analysis:**")
            score_ranges = {
                "90-100": len(df[df['Total'] >= 90]),
                "80-89": len(df[(df['Total'] >= 80) & (df['Total'] < 90)]),
                "70-79": len(df[(df['Total'] >= 70) & (df['Total'] < 80)]),
                "60-69": len(df[(df['Total'] >= 60) & (df['Total'] < 70)]),
                "Below 60": len(df[df['Total'] < 60])
            }
            for range_name, count in score_ranges.items():
                percentage = (count / len(df) * 100) if len(df) > 0 else 0
                st.write(f"- {range_name}: {count} ({percentage:.1f}%)")
        with col3:
            # Answer set effectiveness metrics
            st.markdown("**Answer Set Effectiveness:**")
            avg_accuracy = df['Accuracy'].mean() if 'Accuracy' in df.columns else 0
            flagged_rate = df.get('Flagged', pd.Series(dtype=bool)).sum() / len(df) * 100 if len(df) > 0 else 0
            st.write(f"- Average accuracy: {avg_accuracy:.1f}%")
            st.write(f"- Flagged rate: {flagged_rate:.1f}%")
            st.write(f"- Processing success: 100%")
            # Effectiveness rating
            if avg_accuracy >= 80:
                effectiveness = "Excellent"
                color = "🟢"
            elif avg_accuracy >= 70:
                effectiveness = "Good" 
                color = "🟡"
            else:
                effectiveness = "Needs Review"
                color = "🔴"
            st.write(f"- {color} Effectiveness: {effectiveness}")
    
    # Enhanced score and subject analysis
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Score Distribution")
        if "Total" in df.columns:
            st.bar_chart(df["Total"], height=300)
    with col2:
        st.markdown("### 📚 Subject Performance")
        subject_cols = ["Subject1", "Subject2", "Subject3", "Subject4", "Subject5"]
        if all(col in df.columns for col in subject_cols):
            subject_data = df[subject_cols].mean()
            subject_labels = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5"]
            subject_analysis = pd.DataFrame({
                'Average': subject_data.values,
                'Subject': subject_labels
            })
            st.bar_chart(subject_analysis.set_index('Subject'), height=300)
    
    # Auto-crop performance analytics with answer set correlation
    st.markdown("### 🎯 Auto-Crop Performance Analytics by Answer Set")
    if 'Auto_Cropped' in df.columns:
        cropped_df = df[df['Auto_Cropped'] == True]
        uncropped_df = df[df['Auto_Cropped'] == False]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Auto-Crop vs Full Image Performance:**")
            if not cropped_df.empty and not uncropped_df.empty:
                cropped_avg = cropped_df['Total'].mean()
                uncropped_avg = uncropped_df['Total'].mean()
                performance_data = pd.DataFrame({
                    'Processing Type': ['Auto-Cropped', 'Full Image'],
                    'Average Score': [cropped_avg, uncropped_avg]
                })
                st.bar_chart(performance_data.set_index('Processing Type'))
                # Performance comparison insights
                if cropped_avg > uncropped_avg:
                    improvement = cropped_avg - uncropped_avg
                    st.success(f"Auto-crop improved scores by {improvement:.1f} points on average")
                elif uncropped_avg > cropped_avg:
                    decrease = uncropped_avg - cropped_avg  
                    st.warning(f"Auto-crop decreased scores by {decrease:.1f} points on average")
                else:
                    st.info("Auto-crop and full image processing showed similar performance")
        with col2:
            if not cropped_df.empty and 'Crop_Reduction' in cropped_df.columns:
                st.markdown("**Size Reduction Distribution:**")
                st.bar_chart(cropped_df['Crop_Reduction'])
                avg_reduction = cropped_df['Crop_Reduction'].mean()
                st.info(f"Average size reduction: {avg_reduction:.1%}")
        with col3:
            if not cropped_df.empty and 'Grid_Quality' in cropped_df.columns:
                st.markdown("**Grid Quality Distribution:**")
                st.bar_chart(cropped_df['Grid_Quality'])
                avg_quality = cropped_df['Grid_Quality'].mean()
                quality_rating = "High" if avg_quality > 0.8 else "Medium" if avg_quality > 0.5 else "Low"
                st.info(f"Average quality: {avg_quality:.3f} ({quality_rating})")
    
    # Comprehensive performance metrics
    st.markdown("### 🔬 Comprehensive Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if "Total" in df.columns:
            high_performers = len(df[df["Total"] >= 90])
            st.metric("🏆 High Performers (90+)", f"{high_performers}/{len(df)}")
    with col2:
        if "Auto_Cropped" in df.columns:
            crop_success = len(df[df["Auto_Cropped"] == True])
            crop_rate = crop_success / len(df) * 100 if len(df) > 0 else 0
            st.metric("🎯 Auto-Crop Success", f"{crop_rate:.1f}%")
    with col3:
        if "Selected_Answer_Set" in df.columns:
            answer_set = df["Selected_Answer_Set"].iloc[0]
            st.metric("📋 Answer Set Used", answer_set)
    with col4:
        pipeline_success = len(df) / len(df) * 100 if len(df) > 0 else 0  # All processed files
        st.metric("⚡ Pipeline Success", f"{pipeline_success:.0f}%")
    
    # Answer Set Effectiveness Analysis
    st.markdown("### 📊 Answer Set Effectiveness Analysis")
    if 'Selected_Answer_Set' in df.columns and len(df) > 0:
        answer_set = df['Selected_Answer_Set'].iloc[0]
        # Create effectiveness metrics
        effectiveness_metrics = {
            'Answer Set': answer_set,
            'Files Processed': len(df),
            'Average Score': df['Total'].mean(),
            'Score Std Dev': df['Total'].std(),
            'Auto-Crop Success Rate': len(df[df.get('Auto_Cropped', False)]) / len(df) * 100,
            'High Performers (90+)': len(df[df['Total'] >= 90]) / len(df) * 100,
            'Low Performers (<60)': len(df[df['Total'] < 60]) / len(df) * 100
        }
        # Display as DataFrame
        metrics_df = pd.DataFrame([effectiveness_metrics])
        st.dataframe(metrics_df.round(2), use_container_width=True)
        
        # Insights and recommendations
        st.markdown("### 💡 Insights and Recommendations")
        avg_score = df['Total'].mean()
        score_std = df['Total'].std()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Performance Insights:**")
            if avg_score >= 80:
                st.success(f"✅ Excellent performance with {answer_set}")
            elif avg_score >= 70:
                st.info(f"ℹ️ Good performance with {answer_set}")
            else:
                st.warning(f"⚠️ Performance may need review with {answer_set}")
            if score_std < 10:
                st.info("📊 Consistent scoring across all files")
            else:
                st.warning("📊 High score variation - check individual files")
        with col2:
            st.markdown("**Recommendations:**")
            crop_success_rate = len(df[df.get('Auto_Cropped', False)]) / len(df) * 100
            if crop_success_rate >= 80:
                st.success("🎯 Auto-crop working well - continue using")
            else:
                st.info("🎯 Consider image quality improvements for better auto-crop")
            if len(df[df['Total'] < 60]) > 0:
                st.warning("📋 Review answer set accuracy for low-performing files")
            else:
                st.success("✅ All students performing well with current answer set")

# ========================
# MAIN APP ROUTER
# ========================
def main():
    # Route to appropriate page based on session state
    if st.session_state.current_page == "answer_sets":
        show_answer_sets_page()
    elif st.session_state.current_page == "upload":
        show_upload_page()
    elif st.session_state.current_page == "results":
        show_results_page()
    elif st.session_state.current_page == "comparison":
        show_comparison_page()
    elif st.session_state.current_page == "debug":
        show_debug_page()
    elif st.session_state.current_page == "analytics":
        show_analytics_page()
    else:
        show_answer_sets_page()  # Default to answer sets page

if __name__ == "__main__":
    main()