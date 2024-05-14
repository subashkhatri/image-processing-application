import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def preprocess_image(input_data):
    if input_data.ndim == 3 and input_data.shape[2] == 3:
        hsv = cv2.cvtColor(input_data, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = cv2.equalizeHist(input_data)
    return gray_image

def dynamic_color_segmentation(input_image):
    if len(input_image.shape) == 2:  # Image is grayscale
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))  # Global mean
    h_std, s_std, v_std = np.std(hsv, axis=(0, 1))  # Global standard deviation

    lower_bound = np.array([h_mean - 1.5 * h_std, max(0, s_mean - 1.5 * s_std), max(40, v_mean - v_std)])
    upper_bound = np.array([h_mean + 1.5 * h_std, min(255, s_mean + 1.5 * s_std), 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask

def otsu_thresholding(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_mask

def adaptive_thresholding(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    adaptive_mask = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2
    )
    return adaptive_mask

def combined_segmentation(input_image):
    dynamic_mask = dynamic_color_segmentation(input_image)
    otsu_mask = otsu_thresholding(input_image)
    adaptive_mask = adaptive_thresholding(input_image)

    # Combine the masks using a logical OR operation
    combined_mask = cv2.bitwise_or(dynamic_mask, otsu_mask)
    combined_mask = cv2.bitwise_or(combined_mask, adaptive_mask)

    # Apply morphological closing to fill small holes and connect nearby objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined_mask


def preprocess_image_for_edges(input_data):
    if isinstance(input_data, str):
        image = cv2.imread(input_data)
    else:
        image = input_data
    filtered_image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
    return filtered_image

def hybrid_edge_detection(input_image, low_threshold=50, high_threshold=150):
    if len(input_image.shape) == 3:  # Check if input image has multiple channels
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    elif len(input_image.shape) == 2:  # Input image is already grayscale
        gray_image = input_image
    else:
        raise ValueError("Input image must have either 1 or 3 channels")

    # Use bilateral filter to preserve edges
    filtered_image = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Sobel edge detection with refined kernel size
    sobel_x = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(filtered_image, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))

    # Canny edge detection with adjusted thresholds
    edges_canny = cv2.Canny(filtered_image, low_threshold, high_threshold)

    # Combine Sobel and Canny edges
    combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
    return combined_edges

def apply_morphology(image, kernel_size=(5,5), operation=cv2.MORPH_CLOSE):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    morphed_image = cv2.morphologyEx(image, operation, kernel)
    return morphed_image

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 150]
    return contours

def create_mask(contours, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def calculate_iou(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculate_metrics(predicted_mask, ground_truth_mask):
    predicted_flat = (predicted_mask.flatten() > 127).astype(np.uint8)
    ground_truth_flat = (ground_truth_mask.flatten() > 127).astype(np.uint8)
    precision = precision_score(ground_truth_flat, predicted_flat)
    recall = recall_score(ground_truth_flat, predicted_flat)
    f1 = f1_score(ground_truth_flat, predicted_flat)
    return precision, recall, f1

def calculate_accuracy(predicted_mask, ground_truth_mask):
    correct = np.sum(predicted_mask == ground_truth_mask)
    total = ground_truth_mask.size
    accuracy = correct / total
    return accuracy

def calculate_road_length(contours, pixel_to_real=None):
    total_length = 0
    road_contours = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        road_contours.append(approx)
        segment_length = np.sum(
            [
                np.linalg.norm(approx[i][0] - approx[i - 1][0])
                for i in range(1, len(approx))
            ]
        )
        if pixel_to_real is not None:
            segment_length *= pixel_to_real
        total_length += segment_length
    return total_length, road_contours
