import os
import shutil
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
from string import ascii_lowercase

# Set input and output folder paths
INPUT_FOLDER = r"C:\Users\Admin\.conda\envs\fabric-organizer\data\train\tissu_pattern"
OUTPUT_FOLDER = r"C:\Users\Admin\.conda\envs\fabric-organizer\data\output"

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Function to resize images to 1080x720
def resize_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (1080, 720))
    return img

# Function to compute SIFT descriptors for an image
def get_sift_descriptor(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (1080, 720))  # Ensure consistency
        keypoints, descriptors = sift.detectAndCompute(img, None)
        return descriptors
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return None

# Function to compare SIFT descriptors
def are_images_similar(desc1, desc2, lowe_ratio=0.7, match_threshold=30):
    if desc1 is None or desc2 is None:
        return False

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]

    return len(good_matches) > match_threshold

# Function to calculate SSIM similarity
def calculate_ssim(image1_path, image2_path, threshold=0.8):
    try:
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
        
        img1 = cv2.resize(img1, (1080, 720))
        img2 = cv2.resize(img2, (1080, 720))
        
        score, _ = ssim(img1, img2, full=True)
        return score > threshold
    except Exception as e:
        print(f"⚠️ Error calculating SSIM: {e}")
        return False

# Function to compare color histograms
def histogram_similarity(img1_path, img2_path, threshold=0.85):
    try:
        img1 = resize_image(img1_path)
        img2 = resize_image(img2_path)
        
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity > threshold
    except Exception as e:
        print(f"⚠️ Error comparing histograms: {e}")
        return False

# Load all images and compute SIFT descriptors
image_descriptors = []
image_paths = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        image_path = os.path.join(INPUT_FOLDER, filename)
        descriptors = get_sift_descriptor(image_path)
        if descriptors is not None:
            image_descriptors.append(descriptors)
            image_paths.append(image_path)

# Cluster images by similarity
clusters = defaultdict(list)
cluster_id = 1

for i, (desc1, path1) in enumerate(zip(image_descriptors, image_paths)):
    assigned = False
    for key, group in clusters.items():
        if are_images_similar(desc1, group[0][0]) and calculate_ssim(path1, group[0][1]) and histogram_similarity(path1, group[0][1]):
            clusters[key].append((desc1, path1))
            assigned = True
            break
    if not assigned:
        clusters[cluster_id].append((desc1, path1))
        cluster_id += 1

# Function to generate filenames with letters beyond 'z'
def get_alphabetic_index(i):
    """Convert number to a, b, ... z, aa, ab, ac... style"""
    result = []
    while i >= 0:
        result.append(ascii_lowercase[i % 26])
        i = i // 26 - 1
    return ''.join(result[::-1])

# Organize images into folders and rename them
for idx, (key, images) in enumerate(clusters.items(), start=1):
    folder_name = f"wax_{idx:04d}01"
    folder_path = os.path.join(OUTPUT_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Rename images within the folder
    for i, (_, img_path) in enumerate(images):
        new_name = f"{folder_name}_{get_alphabetic_index(i)}.jpg"
        new_path = os.path.join(folder_path, new_name)
        shutil.copy(img_path, new_path)

print(f"✅ Images successfully organized into {len(clusters)} categories with enhanced pattern matching!")
