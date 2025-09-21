import cv2
import numpy as np
import os
import json
import glob
import pandas as pd
from datetime import datetime
from sklearn.cluster import DBSCAN

# ========================
# AUTO BUBBLE REGION DETECTION AND CROPPING
# ========================
def detect_and_crop_bubble_region(binary_img, original_img, debug=True):
    """
    Automatically detect the main bubble grid area and crop the image
    This removes headers, margins, and focuses only on the answer bubbles
    """
    print("🎯 Step 2.5: Auto-detecting bubble region for cropping...")
    h, w = binary_img.shape
    debug_img = original_img.copy() if debug else None

    # Method 1: Find dense bubble regions using contour analysis
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours that look like bubbles
    bubble_contours = []
    bubble_centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 800:  # Bubble size range
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.2:  # Reasonably circular
                    x, y, w_cont, h_cont = cv2.boundingRect(contour)
                    aspect_ratio = float(w_cont) / h_cont if h_cont > 0 else 0
                    if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio
                        bubble_contours.append(contour)
                        center_x = x + w_cont // 2
                        center_y = y + h_cont // 2
                        bubble_centers.append((center_x, center_y))

    if len(bubble_centers) < 50:
        print("   ⚠️ Insufficient bubbles detected for auto-crop, using full image")
        return None, binary_img, original_img

    # Method 2: Find the bounding region of all bubble centers (initial estimate)
    centers_array = np.array(bubble_centers)
    margin = 50  # Initial margin
    min_x = max(0, np.min(centers_array[:, 0]) - margin)
    max_x = min(w, np.max(centers_array[:, 0]) + margin)
    min_y = max(0, np.min(centers_array[:, 1]) - margin)
    max_y = min(h, np.max(centers_array[:, 1]) + margin)

    # Method 3: Refine boundaries using bubble density
    # Create density map
    density_map = np.zeros((h, w), dtype=np.float32)
    for center_x, center_y in bubble_centers:
        # Add gaussian-like influence around each bubble center
        # Using a smaller radius for potentially sharper density peaks
        influence_radius = 20 # Reduced from 25
        y1, y2 = max(0, center_y - influence_radius), min(h, center_y + influence_radius)
        x1, x2 = max(0, center_x - influence_radius), min(w, center_x + influence_radius)
        density_map[y1:y2, x1:x2] += 1 # Simple increment, could use Gaussian if needed

    # Find the region with highest density - Use a smaller, more focused kernel
    # Smaller kernel size for less smoothing
    kernel_size = max(30, min(w, h) // 20) # Example: adaptive size, min 30px
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    smoothed_density = cv2.filter2D(density_map, -1, kernel)

    
    density_threshold = np.max(smoothed_density) * 0.55
    high_density_mask = (smoothed_density > density_threshold).astype(np.uint8)

    
    coords = cv2.findNonZero(high_density_mask)
    if coords is not None:
        x, y, w_crop, h_crop = cv2.boundingRect(coords)
        
        padding = 20 
        crop_x1 = max(0, x - padding)
        crop_y1 = max(0, y - padding)
        crop_x2 = min(w, x + w_crop + padding)
        crop_y2 = min(h, y + h_crop + padding)
    else:
        
        print("   ⚠️ Density-based crop failed, using bounding box method")
        crop_x1, crop_y1 = int(min_x), int(min_y)
        crop_x2, crop_y2 = int(max_x), int(max_y)

    
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    
    if crop_width < 150 or crop_height < 200:
        print("   ⚠️ Detected crop region too small, using full image")
        return None, binary_img, original_img

    
    total_area = w * h
    crop_area = crop_width * crop_height
    coverage_ratio = crop_area / total_area
    reduction_ratio = 1.0 - coverage_ratio

    
    cropped_binary = binary_img[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_original = original_img[crop_y1:crop_y2, crop_x1:crop_x2]

    
    crop_info = {
        'bbox': (crop_x1, crop_y1, crop_width, crop_height),
        'original_size': (w, h),
        'cropped_size': (crop_width, crop_height),
        'coverage_ratio': coverage_ratio,
        'reduction_ratio': reduction_ratio,
        'bubble_count': len(bubble_centers),
        'density_score': np.mean(smoothed_density[crop_y1:crop_y2, crop_x1:crop_x2]) if crop_area > 0 else 0
    }

    
    if debug:
        cv2.rectangle(debug_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 3)
        cv2.putText(debug_img, f"AUTO-CROP REGION",
                   (crop_x1, crop_y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Reduction: {reduction_ratio:.1%}",
                   (crop_x1, crop_y1-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for center_x, center_y in bubble_centers:
            cv2.circle(debug_img, (center_x, center_y), 3, (255, 0, 0), 1)
        os.makedirs("output/debug_steps", exist_ok=True)
        cv2.imwrite("output/debug_steps/step2_5_auto_crop_detection.jpg", debug_img)
        cv2.imwrite("output/debug_steps/step2_5_cropped_binary.jpg", cropped_binary)
        cv2.imwrite("output/debug_steps/step2_5_cropped_original.jpg", cropped_original)

    print(f"   ✅ Auto-crop successful: {crop_width}x{crop_height}")
    print(f"   📏 Size reduction: {reduction_ratio:.1%} ({w}x{h} → {crop_width}x{crop_height})")
    print(f"   🎯 Bubble density: {len(bubble_centers)} bubbles in region")
    return crop_info, cropped_binary, cropped_original

def detect_bubbles_enhanced_crop(binary_img, original_img, crop_info=None):
    """Enhanced bubble detection optimized for cropped bubble region"""
    print("Step 3: Enhanced bubble detection on cropped region...")
    
    h, w = binary_img.shape
    debug_img = original_img.copy()
    
    
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        
        if 15 < area < 1200 and perimeter > 0:
            x, y, w_cont, h_cont = cv2.boundingRect(contour)
            aspect_ratio = float(w_cont) / h_cont if h_cont > 0 else 0
            
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            extent = area / (w_cont * h_cont) if (w_cont * h_cont) > 0 else 0
            
            
            is_bubble_like = (
                0.3 < aspect_ratio < 3.0 and
                circularity > 0.15 and  # Slightly lower for imperfect circles
                solidity > 0.5 and
                extent > 0.25 and
                area > 15
            )
            
            if is_bubble_like:
                center_x = x + w_cont // 2
                center_y = y + h_cont // 2
                
                
                quality_score = calculate_enhanced_quality_score(
                    area, circularity, aspect_ratio, solidity, extent
                )
                
                bubble_candidates.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, w_cont, h_cont),
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'extent': extent,
                    'quality_score': quality_score,
                    'contour': contour
                })
                
                
                confidence_color = int(255 * quality_score)
                cv2.rectangle(debug_img, (x, y), (x + w_cont, y + h_cont), 
                             (0, confidence_color, 0), 2)
                cv2.circle(debug_img, (center_x, center_y), 2, (255, 0, 0), -1)
                cv2.putText(debug_img, f"{quality_score:.2f}", 
                           (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    
    high_quality_bubbles = [b for b in bubble_candidates if b['quality_score'] > 0.3]
    
    
    high_quality_bubbles.sort(key=lambda x: x['quality_score'], reverse=True)
    
    print(f"   🎯 Enhanced detection: {len(high_quality_bubbles)} high-quality bubbles")
    print(f"   📊 Quality range: {min([b['quality_score'] for b in high_quality_bubbles]):.3f} - {max([b['quality_score'] for b in high_quality_bubbles]):.3f}")
    
    cv2.imwrite("output/debug_steps/step3_enhanced_detection_cropped.jpg", debug_img)
    
    return high_quality_bubbles, debug_img

def calculate_enhanced_quality_score(area, circularity, aspect_ratio, solidity, extent):
    """Calculate enhanced quality score for bubble candidates"""
    
    area_score = min(1.0, area / 300.0)  # Optimal around 300px
    circularity_score = min(1.0, circularity / 0.8)
    aspect_score = 1.0 - abs(aspect_ratio - 1.0)  # Prefer square-ish bubbles
    solidity_score = solidity
    extent_score = extent
    
    
    quality_score = (
        area_score * 0.20 +
        circularity_score * 0.30 +
        aspect_score * 0.20 +
        solidity_score * 0.15 +
        extent_score * 0.15
    )
    
    return min(1.0, quality_score)


def detect_markings_enhanced_crop(binary_img, question_map, original_img, crop_info=None):
    """Enhanced marking detection optimized for cropped bubble regions"""
    print("Step 6: Enhanced marking detection on cropped region...")
    
    answers = [None] * 100
    debug_img = original_img.copy()
    confidence_log = []
    
    
    precision_params = {
        'fill_weight': 0.35,        # Increased weight for fill ratio
        'center_weight': 0.25,      # Center analysis
        'contour_weight': 0.20,     # Contour analysis  
        'gradient_weight': 0.10,    # Gradient analysis
        'texture_weight': 0.10      # Texture analysis
    }
    
    for q_num in range(1, 101):
        if q_num not in question_map:
            continue
        
        question_marks = []
        option_confidences = []
        
        for option in ['A', 'B', 'C', 'D']:
            if option not in question_map[q_num]:
                continue
            
            bubble = question_map[q_num][option]
            
            
            confidence, is_marked = analyze_bubble_enhanced_crop(
                binary_img, bubble, original_img, precision_params
            )
            
            if is_marked:
                question_marks.append(option)
            
            option_confidences.append((option, confidence, is_marked))
            
            
            center = bubble['center']
            bbox = bubble['bbox']
            x, y, w, h = bbox
            
            if is_marked:
                color = (0, 255, 0)  # Green for marked
                thickness = 3
            else:
                color = (100, 100, 255)  # Light blue for unmarked
                thickness = 1
            
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, thickness)
            cv2.circle(debug_img, center, 2, color, -1)
            cv2.putText(debug_img, f"Q{q_num}{option}:{confidence:.2f}", 
                       (x-5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        
        
        final_answer = determine_final_answer_enhanced(
            question_marks, option_confidences, q_num
        )
        answers[q_num - 1] = final_answer
        
        
        confidence_log.append({
            'question': q_num,
            'options': option_confidences,
            'final_answer': final_answer,
            'multiple_marks': len(question_marks) > 1
        })
    
    cv2.imwrite("output/debug_steps/step6_enhanced_marking_detection.jpg", debug_img)
    
    
    save_enhanced_analysis(confidence_log)
    
    marked_count = len([a for a in answers if a])
    print(f"   ✅ Enhanced detection: {marked_count}/100 answers detected")
    
    return answers

def analyze_bubble_enhanced_crop(binary_img, bubble, original_img, precision_params):
    """Enhanced bubble analysis for cropped images"""
    center_x, center_y = bubble['center']
    x, y, w, h = bubble['bbox']
    
    roi_padding = max(1, min(w, h) // 4)
    x1 = max(0, x - roi_padding)
    x2 = min(binary_img.shape[1], x + w + roi_padding)
    y1 = max(0, y - roi_padding)
    y2 = min(binary_img.shape[0], y + h + roi_padding)
    
    roi = binary_img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, False
    
    
    total_pixels = roi.size
    filled_pixels = cv2.countNonZero(roi)
    fill_ratio = filled_pixels / total_pixels
    
    center_size = max(2, min(w, h) // 2)
    cx1 = max(0, center_x - x1 - center_size)
    cx2 = min(roi.shape[1], center_x - x1 + center_size)
    cy1 = max(0, center_y - y1 - center_size)
    cy2 = min(roi.shape[0], center_y - y1 + center_size)
    
    center_fill_ratio = 0
    if cx1 < cx2 and cy1 < cy2:
        center_roi = roi[cy1:cy2, cx1:cx2]
        if center_roi.size > 0:
            center_fill_ratio = cv2.countNonZero(center_roi) / center_roi.size
    
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_score = 0
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        if contour_area > 5:
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                contour_score = (contour_area / (w * h)) * circularity
    
    
    gradient_score = 0
    if y+h <= original_img.shape[0] and x+w <= original_img.shape[1]:
        original_roi = original_img[y:y+h, x:x+w]
        if original_roi.size > 0:
            gray_roi = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY) if len(original_roi.shape) == 3 else original_roi
            gradient_score = np.std(gray_roi) / 255.0
    
    
    texture_score = np.var(roi.astype(np.float32)) / 10000.0 if roi.size > 10 else 0
    
    
    confidence = (
        precision_params['fill_weight'] * min(fill_ratio * 2.5, 1.0) +
        precision_params['center_weight'] * min(center_fill_ratio * 2.0, 1.0) +
        precision_params['contour_weight'] * min(contour_score * 4.0, 1.0) +
        precision_params['gradient_weight'] * gradient_score +
        precision_params['texture_weight'] * min(texture_score, 1.0)
    )
    
    
    base_threshold = 0.25
    quality_bonus = bubble.get('quality_score', 0.5) * 0.05
    adjusted_threshold = base_threshold - quality_bonus
    
    is_marked = (
        confidence > adjusted_threshold and
        (fill_ratio > 0.10 or center_fill_ratio > 0.15) and
        fill_ratio < 0.95  # Reject scanning artifacts
    )
    
    return confidence, is_marked

def determine_final_answer_enhanced(question_marks, option_confidences, q_num):
    """Enhanced answer determination with better conflict resolution"""
    if len(question_marks) == 1:
        return question_marks[0]
    elif len(question_marks) == 0:
        
        sorted_options = sorted(option_confidences, key=lambda x: x[1], reverse=True)
        if sorted_options and sorted_options[0][1] > 0.18:
            print(f"   Q{q_num}: Recovery answer: {sorted_options[0][0]} (conf: {sorted_options[0][1]:.3f})")
            return sorted_options[0][0]
        return None
    else:
        
        marked_options = [opt for opt in option_confidences if opt[0] in question_marks]
        sorted_marks = sorted(marked_options, key=lambda x: x[1], reverse=True)
        
        if len(sorted_marks) >= 2:
            confidence_diff = sorted_marks[0][1] - sorted_marks[1][1]
            if confidence_diff > 0.10:  # Clear winner
                print(f"   Q{q_num}: Resolved conflict: {sorted_marks[0][0]} (diff: {confidence_diff:.3f})")
                return sorted_marks[0][0]
        
        return sorted_marks[0][0] if sorted_marks else None

def save_enhanced_analysis(confidence_log):
    """Save enhanced analysis results"""
    os.makedirs("output", exist_ok=True)
    
    analysis_data = []
    for entry in confidence_log:
        for option, confidence, is_marked in entry['options']:
            analysis_data.append({
                'question': entry['question'],
                'option': option,
                'confidence': confidence,
                'is_marked': is_marked,
                'final_answer': entry['final_answer'],
                'multiple_marks': entry['multiple_marks']
            })
    
    df = pd.DataFrame(analysis_data)
    df.to_csv("output/enhanced_crop_analysis.csv", index=False)
    
    summary = {
        'high_confidence_marks': len(df[(df['is_marked']) & (df['confidence'] > 0.6)]),
        'medium_confidence_marks': len(df[(df['is_marked']) & (df['confidence'] > 0.3) & (df['confidence'] <= 0.6)]),
        'low_confidence_marks': len(df[(df['is_marked']) & (df['confidence'] <= 0.3)]),
        'recovered_marks': len(df[(df['final_answer'] == df['option']) & (~df['is_marked'])]),
        'multiple_mark_questions': len([e for e in confidence_log if e['multiple_marks']]),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    with open("output/enhanced_crop_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def process_single_image_with_autocrop(image_path, selected_answer_set_key, answer_sets, temp_storage=None):
    """Enhanced processing pipeline with automatic bubble region cropping and answer set selection"""
    filename = os.path.basename(image_path)
    print(f"\n🔄 Auto-Crop Enhanced Processing: {filename}")
    print(f"📝 Using Answer Set: {selected_answer_set_key}")
    print("=" * 60)
    
    try:

        gray_img, original_img = convert_to_greyscale(image_path)
        

        binary_img, enhanced_img = apply_enhancement_techniques(gray_img)
        

        crop_info, cropped_binary, cropped_original = detect_and_crop_bubble_region(
            binary_img, original_img, debug=True
        )
        

        if crop_info:
            working_binary = cropped_binary
            working_original = cropped_original
            print(f"   🎯 Using cropped region: {crop_info['cropped_size']}")
        else:
            working_binary = binary_img
            working_original = original_img
            print("   📄 Using full image (no crop detected)")
        

        bubble_candidates, debug_boxes = detect_bubbles_enhanced_crop(
            working_binary, working_original, crop_info
        )
        

        if len(bubble_candidates) < 50:
            print("⚠️ Low bubble count, attempting recovery...")
            working_binary = apply_alternative_preprocessing(
                gray_img if not crop_info else cv2.cvtColor(cropped_original, cv2.COLOR_BGR2GRAY)
            )
            bubble_candidates, debug_boxes = detect_bubbles_enhanced_crop(
                working_binary, working_original, crop_info
            )
        

        grid_info = organize_bubbles_into_grid_enhanced(bubble_candidates)
        
        if not grid_info or grid_info['quality_score'] < 0.3:
            print("⚠️ Grid quality low, attempting fallback...")
            grid_info = create_fallback_grid(bubble_candidates, working_binary.shape)
        

        question_map = map_bubbles_to_questions_enhanced(grid_info, bubble_candidates)
        
        if not question_map or len(question_map) < 80:
            print("⚠️ Insufficient mapping, using coordinate fallback...")
            question_map = create_coordinate_based_mapping(bubble_candidates, working_binary.shape)
        

        extracted_answers = detect_markings_enhanced_crop(
            working_binary, question_map, working_original, crop_info
        )
        

        answer_key = answer_sets.get(selected_answer_set_key, answer_sets.get("SET_A", ["A"] * 100))
        

        subject_scores, total_score, flagged, detailed_results, temp_data = calculate_score_with_comparison(
            extracted_answers, answer_key, temp_storage
        )
        
        accuracy = (total_score / 100) * 100
        flagged_questions = [d['question'] for d in detailed_results if d['is_flagged']]
        
        result = {
            "Filename": filename,
            "Selected_Answer_Set": selected_answer_set_key,
            "Subject1": subject_scores[0],
            "Subject2": subject_scores[1], 
            "Subject3": subject_scores[2],
            "Subject4": subject_scores[3],
            "Subject5": subject_scores[4],
            "Total": total_score,
            "Accuracy": round(accuracy, 2),
            "Flagged": flagged,
            "Flagged_Questions": ",".join(map(str, flagged_questions)),
            "Grid_Quality": grid_info.get('quality_score', 0) if grid_info else 0,
            "Auto_Cropped": crop_info is not None,
            "Crop_Reduction": crop_info['reduction_ratio'] if crop_info else 0,
            "Bubble_Density": crop_info['density_score'] if crop_info else 0,
            "Detection_Method": "Auto-Crop Enhanced Pipeline with Manual Answer Set Selection v2.0"
        }
        
        print(f"✅ Auto-crop processing complete - Score: {total_score}/100 ({accuracy:.1f}%)")
        print(f"📝 Used Answer Set: {selected_answer_set_key}")
        if crop_info:
            print(f"🎯 Crop Reduction: {crop_info['reduction_ratio']:.1%}")
            print(f"📊 Grid Quality: {result['Grid_Quality']:.3f}")
        
        return result, detailed_results
        
    except Exception as e:
        print(f"❌ Error in auto-crop pipeline: {str(e)}")
        return None, []

def organize_bubbles_into_grid_enhanced(bubble_candidates):
    """Enhanced grid organization optimized for cropped regions"""
    print("Step 4: Enhanced grid organization...")
    
    if len(bubble_candidates) < 50:
        return None
    
    centers = np.array([bubble['center'] for bubble in bubble_candidates])
    

    y_coords = centers[:, 1]
    

    y_coords_reshaped = y_coords.reshape(-1, 1)
    clustering = DBSCAN(eps=20, min_samples=3).fit(y_coords_reshaped)
    labels = clustering.labels_
    

    row_clusters = {}
    for i, label in enumerate(labels):
        if label != -1:  # Ignore noise points
            if label not in row_clusters:
                row_clusters[label] = []
            row_clusters[label].append(bubble_candidates[i])
    

    rows = []
    for label in sorted(row_clusters.keys(), key=lambda l: np.mean([b['center'][1] for b in row_clusters[l]])):
        row_bubbles = sorted(row_clusters[label], key=lambda b: b['center'][0])
        if len(row_bubbles) >= 4:  # Need at least 4 bubbles for A,B,C,D
            rows.append(row_bubbles)
    
    total_bubbles = sum(len(row) for row in rows)
    expected_bubbles = len(rows) * 4
    quality_score = min(1.0, total_bubbles / max(expected_bubbles, 1)) if expected_bubbles > 0 else 0
    
    
    row_consistency = np.std([len(row) for row in rows]) if rows else 0
    quality_adjustment = max(0, 1.0 - (row_consistency / 10.0))
    final_quality = quality_score * quality_adjustment
    
    print(f"   Enhanced grid: {len(rows)} rows, {total_bubbles} bubbles, quality: {final_quality:.3f}")
    
    grid_info = {
        'rows': rows,
        'total_rows': len(rows),
        'total_bubbles': total_bubbles,
        'quality_score': final_quality,
        'row_consistency': row_consistency
    }
    
    return grid_info

def map_bubbles_to_questions_enhanced(grid_info, bubble_candidates):
    """Enhanced question mapping with better spacing analysis"""
    print("Step 5: Enhanced question mapping...")
    
    if not grid_info:
        return {}
    
    rows = grid_info['rows']
    question_map = {}
    
    question_num = 1
    for row in rows:
        if len(row) < 4:
            continue
        
        question_groups = []
        current_group = []
        last_x = -1
        
        for bubble in row:
            x = bubble['center'][0]
            
        
            if last_x != -1 and x - last_x > 80:  # Adjustable spacing threshold
                if len(current_group) >= 4:
                    question_groups.append(current_group)
                current_group = [bubble]
            else:
                current_group.append(bubble)
            
            last_x = x
        
        
        if len(current_group) >= 4:
            question_groups.append(current_group)
        
        
        if not question_groups:
            for i in range(0, len(row), 4):
                if i + 3 < len(row):
                    question_groups.append(row[i:i+4])
        
        
        for group in question_groups:
            if question_num > 100:
                break
            
            if len(group) >= 4:
                question_map[question_num] = {}
                for j, bubble in enumerate(group[:4]):
                    option = ['A', 'B', 'C', 'D'][j]
                    question_map[question_num][option] = bubble
                question_num += 1
    
    print(f"   Enhanced mapping: {len(question_map)} questions mapped")
    return question_map


def convert_to_greyscale(image_path):
    """Load image and convert to greyscale"""
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Cannot load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    os.makedirs("output/debug_steps", exist_ok=True)
    cv2.imwrite("output/debug_steps/step1_greyscale.jpg", gray)
    
    print("Step 1: Greyscale conversion completed")
    return gray, img

def apply_enhancement_techniques(gray_img):
    """Apply image enhancement for better bubble detection"""
    print("Step 2: Applying enhancement techniques...")
    

    denoised = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)
    

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    

    filtered = cv2.bilateralFilter(enhanced, 9, 80, 80)
    

    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite("output/debug_steps/step2_enhanced.jpg", binary)
    print("   Enhancement completed")
    
    return binary, enhanced

def apply_alternative_preprocessing(gray_img):
    """Alternative preprocessing for difficult images"""
    print("   Applying alternative preprocessing...")
    
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_img)
    
    
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 4)
    
    cv2.imwrite("output/debug_steps/step2_alternative.jpg", binary)
    return binary

def create_fallback_grid(bubble_candidates, image_shape):
    """Create fallback grid structure"""
    return {
        'rows': [bubble_candidates[i:i+20] for i in range(0, len(bubble_candidates), 20)],
        'total_rows': max(1, len(bubble_candidates) // 20),
        'total_bubbles': len(bubble_candidates),
        'quality_score': 0.4
    }

def create_coordinate_based_mapping(bubble_candidates, image_shape):
    """Create coordinate-based mapping as fallback"""
    question_map = {}
    question_num = 1
    
    
    sorted_bubbles = sorted(bubble_candidates, key=lambda b: (b['center'][1], b['center'][0]))
    
    
    for i in range(0, len(sorted_bubbles) - 3, 4):
        if question_num > 100:
            break
        
        question_bubbles = sorted_bubbles[i:i+4]
        question_map[question_num] = {}
        
        for j, bubble in enumerate(question_bubbles):
            option = ['A', 'B', 'C', 'D'][j]
            question_map[question_num][option] = bubble
        
        question_num += 1
    
    return question_map

def create_answer_sets_input():
    """Interactive function to create answer sets A and B"""
    print("\n" + "="*60)
    print("ANSWER SET CONFIGURATION")
    print("="*60)
    print("Please provide answers for 100 questions (A, B, C, or D)")
    print("Format: Enter answers as a continuous string or comma-separated")
    print("Example: ABCDABCD... or A,B,C,D,A,B,C,D...")
    print("="*60)
    
    answer_sets = {}
    
    for set_name in ['A', 'B']:
        print(f"\nCreating Answer Set {set_name}:")
        print("-" * 40)
        
        while True:
            try:
                user_input = input(f"Enter 100 answers for Set {set_name}: ").strip().upper()
                
                # Parse input
                if ',' in user_input:
                    answers = [ans.strip() for ans in user_input.split(',')]
                else:
                    answers = list(user_input.replace(' ', ''))
                
                # Validate
                if len(answers) != 100:
                    print(f"Error: Expected 100 answers, got {len(answers)}. Please try again.")
                    continue
                
                valid_answers = all(ans in ['A', 'B', 'C', 'D'] for ans in answers)
                if not valid_answers:
                    print("Error: Only A, B, C, D are allowed. Please try again.")
                    continue
                
                answer_sets[f"SET_{set_name}"] = answers
                print(f"Set {set_name} created successfully!")
                break
                
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return None
            except Exception as e:
                print(f"Error: {str(e)}. Please try again.")
    
    return answer_sets

def load_or_create_answer_sets(answer_keys_path="answer_keys.json"):
    """Load existing answer sets or create new ones"""
    print("\nANSWER SET MANAGEMENT")
    print("=" * 50)
    
    if os.path.exists(answer_keys_path):
        try:
            with open(answer_keys_path, 'r') as f:
                existing_sets = json.load(f)
            
            if "SET_A" in existing_sets and "SET_B" in existing_sets:
                print("Found existing answer sets A and B")
                choice = input("Use existing sets? (Y/n): ").strip().lower()
                if choice in ['', 'y', 'yes']:
                    return existing_sets
        except Exception as e:
            print(f"Error reading existing file: {e}")
    

    print("Creating new answer sets...")
    new_sets = create_answer_sets_input()
    
    if new_sets:
        try:
            with open(answer_keys_path, 'w') as f:
                json.dump(new_sets, f, indent=2)
            print(f"Answer sets saved to: {answer_keys_path}")
        except Exception as e:
            print(f"Error saving: {e}")
    
    return new_sets

def select_answer_set_for_processing(answer_sets):
    """Allow user to select which answer set to use for processing"""
    available_sets = list(answer_sets.keys())
    
    print(f"\nAVAILABLE ANSWER SETS FOR PROCESSING:")
    print("=" * 50)
    
    for i, set_key in enumerate(available_sets, 1):
        preview = ''.join(answer_sets[set_key][:10]) + "..."
        print(f"{i}. {set_key}: {preview}")
    
    while True:
        try:
            choice = input(f"\nSelect answer set (1-{len(available_sets)}): ").strip()
            
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_sets):
                    selected_key = available_sets[choice_idx]
                    print(f"Selected: {selected_key}")
                    return selected_key
                else:
                    print(f"Please enter a number between 1 and {len(available_sets)}")
            else:
                print("Please enter a valid number")
                
        except KeyboardInterrupt:
            print("\nUsing default SET_A")
            return "SET_A"
        except Exception as e:
            print(f"Error: {e}. Please try again.")

def calculate_score_with_comparison(extracted_answers, answer_key, temp_storage=None):
    """Score answers with direct array comparison"""
    print("\nANSWER COMPARISON PROCESS")
    print("=" * 50)
    
    if temp_storage is None:
        temp_storage = []
    
    temp_storage.clear()
    temp_storage.extend(extracted_answers)
    
    # Ensure proper lengths
    while len(temp_storage) < 100:
        temp_storage.append(None)
    temp_storage = temp_storage[:100]
    
    while len(answer_key) < 100:
        answer_key.extend(answer_key[:min(4, 100-len(answer_key))])
    answer_key = answer_key[:100]
    

    subject_scores = [0] * 5
    detailed_results = []
    
    for i in range(100):
        subject_idx = i // 20
        question_num = i + 1
        
        student_answer = temp_storage[i]
        correct_answer = answer_key[i]
        
        is_correct = (student_answer == correct_answer and student_answer is not None)
        is_flagged = student_answer is None
        
        if is_correct:
            subject_scores[subject_idx] += 1
        
        detailed_results.append({
            'question': question_num,
            'subject': subject_idx + 1,
            'temp_answer': student_answer,
            'key_answer': correct_answer,
            'is_correct': is_correct,
            'is_flagged': is_flagged
        })
    
    total_score = sum(subject_scores)
    overall_flagged = len([d for d in detailed_results if d['is_flagged']]) > 15
    
    print(f"Total Score: {total_score}/100 ({(total_score/100)*100:.1f}%)")
    
    return subject_scores, total_score, overall_flagged, detailed_results, temp_storage


def batch_process_with_autocrop_and_selection(input_folder="input", output_csv="output/autocrop_results.csv"):
    """Batch process images with auto-crop feature and answer set selection"""
    print("AUTO-CROP ENHANCED OMR PROCESSING WITH ANSWER SET SELECTION")
    print("=" * 80)
    

    answer_sets = load_or_create_answer_sets()
    if not answer_sets:
        print("No answer sets available. Exiting.")
        return pd.DataFrame()
    

    selected_answer_set_key = select_answer_set_for_processing(answer_sets)
    

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {input_folder}")
        return pd.DataFrame()
    
    print(f"Found {len(image_paths)} images for processing with answer set: {selected_answer_set_key}")
    
    results = []
    temp_storage = []
    
    for i, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        print(f"\nProcessing [{i}/{len(image_paths)}]: {filename}")
        
        try:
            result, detailed_results = process_single_image_with_autocrop(
                img_path, selected_answer_set_key, answer_sets, temp_storage
            )
            
            if result:
                results.append(result)
                crop_status = "CROPPED" if result.get('Auto_Cropped', False) else "FULL"
                print(f"Success - Score: {result['Total']}/100 ({crop_status}) using {selected_answer_set_key}")
            else:
                print(f"Failed to process {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    

    if results:
        df = pd.DataFrame(results)
        os.makedirs("output", exist_ok=True)
        df.to_csv(output_csv, index=False)
        

        cropped_count = len(df[df.get('Auto_Cropped', pd.Series([False] * len(df)))])
        avg_reduction = df[df.get('Auto_Cropped', pd.Series([False] * len(df)))].get('Crop_Reduction', pd.Series([0])).mean()
        
        print(f"\nProcessing Complete with {selected_answer_set_key}!")
        print(f"Success Rate: {len(results)}/{len(image_paths)}")
        print(f"Average Score: {df['Total'].mean():.1f}/100")
        print(f"Auto-Cropped: {cropped_count}/{len(results)} ({cropped_count/len(results)*100:.1f}%)")
        if cropped_count > 0:
            print(f"Average Size Reduction: {avg_reduction:.1%}")
        print(f"Results saved to: {output_csv}")
    
    return df if results else pd.DataFrame()


if __name__ == "__main__":
    print("AUTO-CROP ENHANCED OMR PROCESSING WITH ANSWER SET SELECTION v2.0")
    print("=" * 80)
    print("Features:")
    print("- Automatic bubble region detection and cropping")
    print("- Enhanced precision on cropped bubble area")
    print("- User-defined answer sets (A & B)")
    print("- Manual answer set selection for processing")
    print("- Direct array comparison scoring")
    print("=" * 80)
    
    results_df = batch_process_with_autocrop_and_selection("input", "output/autocrop_results.csv")
    
    if not results_df.empty:
        print(f"\nProcessing Summary:")
        print(f"Files processed: {len(results_df)}")
        print(f"Average accuracy: {results_df['Accuracy'].mean():.2f}%")
        
        if 'Auto_Cropped' in results_df.columns:
            cropped_files = results_df[results_df['Auto_Cropped']]
            if len(cropped_files) > 0:
                print(f"Auto-cropped files: {len(cropped_files)}")
                print(f"Average size reduction: {cropped_files['Crop_Reduction'].mean():.1%}")
        
        
        if 'Selected_Answer_Set' in results_df.columns:
            used_set = results_df['Selected_Answer_Set'].iloc[0]
            print(f"Answer set used: {used_set}")
    
    print("\nAuto-Crop Enhanced OMR Processing with Answer Set Selection Complete!")