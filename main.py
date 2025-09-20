import cv2
import numpy as np
import os
import json
import sqlite3
from datetime import datetime
import glob
import pandas as pd

# ========================
# IMAGE PREPROCESSING
# ========================
def preprocess_image(image_path):
    """Load, crop, enhance, and threshold OMR sheet"""
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Image not found: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- MANUAL CROP ---
    h, w = img.shape[:2]
    y_start = 150   # Adjust based on your sheet
    y_end = h - 50
    x_start = 50
    x_end = w - 50

    cropped_img = img[y_start:y_end, x_start:x_end]
    cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    blurred = cv2.GaussianBlur(cropped_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=0.02, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)

    # Threshold (filled = dark)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return final_thresh, cropped_img

# ========================
# BUBBLE DETECTION (4 OPTIONS: A/B/C/D)
# ========================
def extract_answers(binary_img, colored_img=None):
    """Extract answers from 4-option bubble grid"""
    h, w = binary_img.shape
    answers = []
    debug_img = colored_img.copy() if colored_img is not None else None

    # Grid parameters — ADJUST FOR YOUR SHEET
    start_y = 80    # First bubble row
    start_x = 120   # Left of first option (A)
    dy = 30         # Vertical spacing
    dx = 60         # Horizontal spacing (A→B→C→D)
    roi_size = 40

    for q in range(100):  # 100 questions
        row = q // 4      # 4 options per question
        col = q % 4       # 0=A, 1=B, 2=C, 3=D

        section = q // 20  # 20 questions per subject
        y = start_y + section * (20 * dy + 50) + row * dy
        x = start_x + col * dx

        y1, y2 = int(y - roi_size//2), int(y + roi_size//2)
        x1, x2 = int(x - roi_size//2), int(x + roi_size//2)

        if y1 < 0 or y2 > h or x1 < 0 or x2 > w:
            answers.append(None)
            continue

        roi = binary_img[y1:y2, x1:x2]
        fill_ratio = cv2.countNonZero(roi) / roi.size

        # Debug print
        print(f"Q{q+1} {['A','B','C','D'][col]}: fill_ratio = {fill_ratio:.2f}")

        option = None
        if fill_ratio > 0.15:  # Threshold for marking
            option = ["A", "B", "C", "D"][col]

        answers.append(option)

        # Draw debug rectangles
        if debug_img is not None:
            color = (0, 255, 0) if option else (0, 0, 255)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

    if debug_img is not None:
        cv2.imwrite("output/debug_bubbles.jpg", debug_img)

    return answers

# ========================
# VERSION DETECTION (SET A/B/C/D)
# ========================
def detect_version(binary_img):
    """Detect version from top-right 4 bubbles"""
    h, w = binary_img.shape

    # Version bubble area — ADJUST FOR YOUR SHEET
    version_start_x = w - 250  # 250px from right
    version_start_y = 50       # 50px from top
    version_dx = 60            # spacing between bubbles
    roi_size = 40

    version_options = ["SET_A", "SET_B", "SET_C", "SET_D"]
    detected_version = "SET_A"
    max_fill = 0

    for i in range(4):
        x = version_start_x + i * version_dx
        y = version_start_y

        y1, y2 = int(y - roi_size//2), int(y + roi_size//2)
        x1, x2 = int(x - roi_size//2), int(x + roi_size//2)

        if y1 < 0 or y2 > h or x1 < 0 or x2 > w:
            continue

        roi = binary_img[y1:y2, x1:x2]
        fill_ratio = cv2.countNonZero(roi) / roi.size

        print(f"Version {version_options[i]} bubble: fill_ratio = {fill_ratio:.2f}")

        if fill_ratio > 0.15 and fill_ratio > max_fill:
            max_fill = fill_ratio
            detected_version = version_options[i]

    print(f"🔖 Auto-detected Version: {detected_version}")
    return detected_version

# ========================
# EXAM CONFIG LOADER
# ========================
def load_exam_config(exam_config_path):
    """Load exam config including answer key and metadata"""
    with open(exam_config_path, "r") as f:
        return json.load(f)

# ========================
# SCORING ENGINE
# ========================
def calculate_score(extracted_answers, exam_config):
    """Calculate subject-wise and total score"""
    answer_key = exam_config["answer_key"]
    if len(extracted_answers) != 100:
        raise Exception(f"Expected 100 extracted answers, got {len(extracted_answers)}")
    if len(answer_key) != 100:
        raise Exception(f"Expected 100 answer key items, got {len(answer_key)}")

    correct = [0] * 5  # 5 subjects
    flagged = False

    for i in range(100):
        subject_idx = i // 20
        if extracted_answers[i] == answer_key[i]:
            correct[subject_idx] += 1

    total = sum(correct)
    return correct, total, flagged

# ========================
# DATABASE & AUDIT TRAIL
# ========================
def init_db(db_path="output/omr_results.db"):
    """Initialize SQLite database"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            exam_id TEXT,
            version TEXT,
            subject1 INTEGER,
            subject2 INTEGER,
            subject3 INTEGER,
            subject4 INTEGER,
            subject5 INTEGER,
            total INTEGER,
            flagged BOOLEAN,
            processed_at TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_result_to_db(result, db_path="output/omr_results.db"):
    """Save result to database"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO results 
        (filename, exam_id, version, subject1, subject2, subject3, subject4, subject5, total, flagged, processed_at, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result["Filename"],
        result.get("ExamID", "UNKNOWN"),
        result["Version"],
        result["Subject1"],
        result["Subject2"],
        result["Subject3"],
        result["Subject4"],
        result["Subject5"],
        result["Total"],
        result["Flagged"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        result.get("Debug_Image", "")
    ))
    conn.commit()
    conn.close()

# ========================
# BATCH PROCESSING
# ========================
def batch_process(input_folder="input", output_csv="output/results.csv"):
    """Process all images in input folder"""
    image_paths = (glob.glob(os.path.join(input_folder, "*.jpg")) +
                   glob.glob(os.path.join(input_folder, "*.jpeg")) +
                   glob.glob(os.path.join(input_folder, "*.png")))

    if not image_paths:
        print("❌ No images found in /input")
        return

    init_db()  # Initialize DB
    results = []

    for img_path in image_paths:
        try:
            filename = os.path.basename(img_path)
            print(f"\n📄 Processing: {filename}")

            # Preprocess
            processed_img, rectified_img = preprocess_image(img_path)

            # Extract answers
            answers = extract_answers(processed_img, rectified_img)

            # Detect version
            version = detect_version(processed_img)

            # Load exam config
            exam_config_path = f"exams/math_exam_{version.lower()}.json"
            if not os.path.exists(exam_config_path):
                raise Exception(f"Exam config not found: {exam_config_path}")

            exam_config = load_exam_config(exam_config_path)

            # Calculate score
            subject_scores, total_score, flagged = calculate_score(answers, exam_config)

            # Save debug image
            debug_path = os.path.join("output", f"debug_{filename}")
            if os.path.exists("output/debug_bubbles.jpg"):
                cv2.imwrite(debug_path, cv2.imread("output/debug_bubbles.jpg"))

            # Prepare result
            result = {
                "Filename": filename,
                "ExamID": exam_config["exam_id"],
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

            # Save to DB
            save_result_to_db(result)

            results.append(result)
            print(f"✅ Score: {total_score}/100")

        except Exception as e:
            print(f"❌ Error processing {img_path}: {str(e)}")

    # Export to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n🎉 Batch processing complete! Results saved to: {output_csv}")
    return df

# ========================
# SINGLE FILE PROCESSING (for testing)
# ========================
if __name__ == "__main__":
    print("🚀 Starting OMR Pipeline...")

    # OPTION 1: Process single file
    # processed_img, rectified_img = preprocess_image("input/Img1.jpeg")
    # answers = extract_answers(processed_img, rectified_img)
    # version = detect_version(processed_img)
    # exam_config = load_exam_config(f"exams/math_exam_{version.lower()}.json")
    # subject_scores, total_score, flagged = calculate_score(answers, exam_config)
    # print(f"✅ Subject Scores: {subject_scores}")
    # print(f"💯 Total Score: {total_score}/100")

    # OPTION 2: Process ALL files (recommended)
    batch_process("input", "output/results.csv")