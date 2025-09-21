## 🎯 Automated OMR Evaluation & Scoring System

An advanced Optical Mark Recognition (OMR) system designed to automatically detect, crop, and evaluate answer sheets with enhanced accuracy. This system features intelligent bubble region detection, user-defined answer sets, and a comprehensive web interface for processing and analysis.

**🚀 Live Deployment:** [https://omr-evaluator.streamlit.app/](https://omr-evaluator.streamlit.app/)

## ✨ Key Features

*   **🎯 Automatic Bubble Region Detection & Cropping:** Intelligently identifies the main answer grid area, removing headers and margins for focused processing.
*   **🔍 Enhanced Precision on Cropped Region:** Applies optimized algorithms specifically to the detected bubble area for improved accuracy.
*   **📝 Flexible Answer Set Management:** Create, manage, and select from multiple predefined answer sets (e.g., Set A, Set B) for scoring.
*   **⚡ Streamlit Web Interface:** User-friendly GUI for uploading images, configuring answer sets, viewing results, and debugging.
*   **📊 Detailed Results & Analytics:** Provides scores, accuracy metrics, flagged questions, and comparative analysis.
*   **🔍 Visual Debugging Pipeline:** Inspect intermediate processing steps to understand and troubleshoot the system's behavior.


## 🚀 Getting Started

**Prerequisites**

   Python 3.7+
   Required Python libraries (install via `pip install -r requirements.txt`):

    ```bash
    opencv-python
    numpy
    scikit-learn
    pandas
    streamlit
    ```

## Installation

1.  Clone or download this repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure `main.py` (containing the core processing logic) is in the same directory as `app.py` (the Streamlit application).


## Running the Application

1.  Open a terminal or command prompt in the project directory.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  Your default web browser should open the application. If not, navigate to the URL provided in the terminal (usually `http://localhost:8501`).



## 🧭 Using the System

1.  **📝 Answer Sets Management:**
       Navigate to the "Answer Sets" page in the sidebar.
       Create or upload your answer keys (100 answers: A, B, C, D). You can manage multiple sets.
       Save your sets for use in processing.

2.  **🎯 Image Processing:**
       Go to the "Image Processing" page.
       Select the desired answer set from the dropdown.
       Upload one or more scanned OMR sheet images (JPG, PNG, TIFF, BMP).
       Click "Start Processing with Selected Answer Set".
       The system will automatically detect the bubble region, crop the image, analyze the marks, and score the sheets based on the chosen answer set.

3.  **📊 View Results:**
       After processing, navigate to the "View Results" page to see a summary table of scores and accuracy for each processed sheet.

4.  **🔍 Answer Comparison:**
       The "Answer Comparison" page offers a detailed, question-by-question breakdown of the student's answers versus the selected answer key, including filtering and subject-wise analysis.

5.  **🔍 Debug Pipeline:**
       Use the "Debug Pipeline" page to view intermediate images generated during the processing of the last batch, helping to visualize how the system interprets the sheets.

6.  **📈 Analytics:**
       The "Analytics" page provides performance charts, score distributions, subject-wise performance, and insights into the effectiveness of the auto-crop feature and the selected answer set.

## 📁 Project Structure

*   `app.py`: Main Streamlit application file providing the web interface.
*   `main.py`: Core Python script containing the OMR processing logic (auto-crop, detection, scoring). *(Content from Pasted_Text_1758424614496.txt)*
*   `requirements.txt`: List of required Python packages.
*   `input/`: Folder where uploaded images are temporarily stored during processing.
*   `output/`: Folder where results (CSVs), debug images, and analysis files are saved.
    *   `output/debug_steps/`: Intermediate processing images for debugging.
    *   `output/autocrop_results.csv`: Main results file generated after batch processing.
    *   `output/enhanced_crop_analysis.csv`: Detailed bubble marking confidence data.
    *  `output/enhanced_crop_summary.json`: Summary statistics of the marking detection.
*   `answer_keys.json`: File storing the created answer sets in JSON format.

## 🛠️ How It Works (High-Level)

1.  **Preprocessing:** The scanned image is converted to greyscale and enhanced (denoising, contrast adjustment, filtering).
2.  **Auto-Crop Detection:** Contours are analyzed to find potential bubbles. A density map identifies the main cluster, and the image is cropped to this region.
3.  **Bubble Detection (Enhanced):** Contours within the cropped region are analyzed for size, shape, and quality to identify valid answer bubbles.
4.  **Grid Organization:** Identified bubbles are clustered into rows using DBSCAN and grouped into questions based on spacing.
5.  **Marking Detection (Enhanced):** For each bubble, multiple metrics (fill ratio, center density, contour shape, etc.) are calculated to determine if it's marked. A weighted confidence score is generated.
6.  **Answer Mapping & Scoring:** Bubbles are mapped to questions (Q1A, Q1B...). The detected answers are compared against the user-selected answer set. Scores are calculated per subject and overall.
7.  **Results & Analysis:** Scores, accuracy, flagged questions, and detailed comparison data are generated and made available through the web interface.

## 🧠 Technical Deep Dive

### 🎯 Auto-Crop Mechanism (`detect_and_crop_bubble_region`)

This function intelligently finds the region of the image containing the answer bubbles:

1.  **Contour Analysis:** Finds external contours in the binary image and filters them based on area (20-800 pixels), circularity (>0.2), and aspect ratio (0.3-3.0) to identify potential bubbles.
2.  **Density Map Creation:** A density map is generated by adding a small Gaussian influence around each detected bubble center.
3.  **Smoothing & Thresholding:** The density map is smoothed with a large kernel (100x100) to find the core bubble cluster area. A threshold (30% of max density) is applied.
4.  **Bounding Box & Cropping:** The bounding box of the high-density mask is found, padded slightly, validated for minimum size, and used to crop both the binary and original images.
5.  **Debugging:** Visualizes the detected region and bubbles on the original image and saves intermediate results.

### 🔍 Enhanced Bubble Detection (`detect_bubbles_enhanced_crop`)

This function focuses on finding high-quality bubbles within the (potentially cropped) image:

1.  **Contour Finding:** Finds external contours in the working binary image.
2.  **Enhanced Filtering:** Filters contours based on size (15-1200 pixels), perimeter, aspect ratio (0.3-3.0), circularity (>0.15), solidity (>0.5), and extent (>0.25). These parameters are tuned for the cropped region.
3.  **Quality Scoring:** Assigns a `quality_score` to each candidate based on a weighted combination of normalized area, circularity, aspect ratio, solidity, and extent.
4.  **Selection:** Selects candidates with a `quality_score` > 0.3 and sorts them.
5.  **Debugging:** Visualizes detected bubbles with bounding boxes and quality scores.

### 🧮 Enhanced Quality Score (`calculate_enhanced_quality_score`)

Calculates a single quality metric for a bubble candidate:
*   **Area Score:** Prefers bubbles around 300px.
*   **Circularity Score:** Prefers more circular bubbles.
*   **Aspect Ratio Score:** Prefers square-ish bubbles.
*   **Solidity & Extent Scores:** Prefer bubbles that are solid and fill their bounding box well.
*   These are combined with weights: Area (20%), Circularity (30%), Aspect Ratio (20%), Solidity (15%), Extent (15%).

### 📐 Grid Organization (`organize_bubbles_into_grid_enhanced`)

Organizes detected bubbles into rows and questions:

1.  **Row Clustering (DBSCAN):** Uses DBSCAN clustering on the Y-coordinates of bubble centers to group them into horizontal rows, ignoring noise.
2.  **Row Sorting & Filtering:** Sorts rows by average Y-coordinate and discards rows with fewer than 4 bubbles.
3.  **Quality Calculation:** Calculates a `quality_score` based on the ratio of total bubbles found to the expected number (rows * 4), adjusted by row consistency (standard deviation of row lengths).

### 🗺️ Question Mapping (`map_bubbles_to_questions_enhanced`)

Maps organized rows of bubbles to specific questions and options (A, B, C, D):

1.  **Iterate Rows:** Goes through each identified row.
2.  **Question Grouping:** Within a row, groups bubbles into questions by looking for significant horizontal gaps (>80 pixels) between them.
3.  **Fallback Grouping:** If no clear gaps, it falls back to grouping every 4 consecutive bubbles as a question.
4.  **Mapping:** Assigns the first 4 bubbles in each group to options A, B, C, D for questions 1 through 100.

### ✅ Enhanced Marking Detection (`detect_markings_enhanced_crop`)

Determines if a bubble is marked and calculates confidence:

1.  **Iterate Questions:** Goes through questions 1-100.
2.  **Iterate Options:** For each option (A, B, C, D) of a question, analyzes the corresponding bubble.
3.  **Bubble Analysis (`analyze_bubble_enhanced_crop`):**
    *   **ROI:** Defines a Region of Interest (ROI) around the bubble, including padding.
    *   **Fill Ratio:** Calculates the ratio of filled pixels in the ROI.
    *   **Center Fill Ratio:** Calculates the fill ratio specifically in the center of the bubble.
    *   **Contour Analysis:** Finds the largest contour in the ROI and calculates a score based on its area and circularity.
    *   **Gradient/Texture Analysis:** Calculates standard deviation (gradient) and variance (texture) within the ROI as additional features.
    *   **Confidence Calculation:** Combines these metrics with weights: Fill (35%), Center (25%), Contour (20%), Gradient (10%), Texture (10%).
    *   **Marking Decision:** Determines if marked based on the confidence score (adjusted threshold), fill ratio, and center fill ratio, while rejecting very high fill (artifacts).
4.  **Answer Determination (`determine_final_answer_enhanced`):**
    *   If exactly one option is marked, that's the answer.
    *   If no options are marked, it attempts recovery by selecting the option with the highest confidence (>0.18).
    *   If multiple options are marked, it selects the one with the highest confidence, or resolves conflicts if the confidence difference is significant (>0.10).
5.  **Logging:** Saves detailed analysis and a summary to CSV/JSON files.

### 📊 Answer Set Management & Scoring

*   **Answer Sets:** Stored in `answer_keys.json`. Users can create multiple sets.
*   **Selection:** Users select which answer set to use before processing a batch.
*   **Scoring (`calculate_score_with_comparison`):** Compares the extracted answers directly against the selected answer key. Calculates subject-wise scores (20 questions per subject) and a total score. Flags sheets with many unanswered questions.

## 📈 Streamlit Web Application (`app.py`)

The `app.py` file provides a user-friendly interface for interacting with the OMR system:

*   **Navigation Sidebar:** Allows switching between different sections of the app.
*   **Answer Sets Page:**
    *   View existing answer sets.
    *   Create new sets via an interactive form or by uploading JSON/Excel/CSV files.
    *   Download or delete answer sets.
*   **Image Processing Page:**
    *   Select an answer set for the current processing session.
    *   Upload one or more OMR sheet images.
    *   Initiates batch processing using the `process_single_image_with_autocrop` function.
    *   Displays real-time progress and results for each file, including score, auto-crop status, and size reduction.
    *   Stores results in `st.session_state` for use by other pages.
*   **View Results Page:**
    *   Displays a table of results from the last processing batch.
    *   Shows key metrics like total sheets, average score, auto-crop success, and flagged sheets.
    *   Allows downloading the results CSV.
*   **Answer Comparison Page:**
    *   Provides a detailed, question-by-question comparison.
    *   Includes filtering options (correct/incorrect/unanswered, subject).
    *   Offers subject-wise performance analysis.
    *   Allows downloading filtered and full comparison data.
*   **Debug Pipeline Page:**
    *   Displays intermediate images generated during the last processing batch (e.g., greyscale, enhanced, cropped, detected bubbles, markings).
    *   Provides descriptions for each processing step.
    *   Shows auto-crop specific metrics.
*   **Analytics Page:**
    *   Presents charts for score distribution and subject performance.
    *   Analyzes auto-crop performance (success rate, size reduction, grid quality).
    *   Provides comprehensive performance metrics and insights related to the selected answer set's effectiveness.
