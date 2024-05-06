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
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    lower_bound = np.array([h_mean - 10, s_mean - 40, 40])
    upper_bound = np.array([h_mean + 10, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask


def adaptive_color_segmentation(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]
    adaptive_thresh = cv2.adaptiveThreshold(
        value_channel,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9,
    )
    return adaptive_thresh


def preprocess_image_for_edges(input_data):
    if isinstance(input_data, str):
        image = cv2.imread(input_data)
    else:
        image = input_data
    filtered_image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
    return filtered_image


def hybrid_edge_detection(input_image, low_threshold=50, high_threshold=150, sobel_kernel_size=3):
    if len(input_image.shape) > 2:
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = input_image

    # Sobel edge detection
    edges_sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    edges_sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    edges_sobel = cv2.convertScaleAbs(cv2.magnitude(edges_sobel_x, edges_sobel_y))

    # Canny edge detection
    edges_canny = cv2.Canny(gray_image, low_threshold, high_threshold)

    # Combine all edge maps using bitwise operations
    combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)

    return combined_edges

def advanced_edge_detection(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    v = np.median(blurred)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)
    return edges

def combined_edge_detection(image):
    # Get initial edges using advanced Canny method
    initial_edges = advanced_edge_detection(image)

    # Refine edges using the hybrid method
    refined_edges = hybrid_edge_detection(image)

    # Combine the two edge maps
    combined_edges = cv2.bitwise_or(initial_edges, refined_edges)
    return combined_edges


def apply_morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing


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
