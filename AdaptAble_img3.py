import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import os

def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def load_image(image_path):
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    return image, original_size

def detect_objects(image, model, target_classes):
    results = model(image)
    detections = results.xyxy[0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for *box, conf, cls in detections:
        class_name = model.names[int(cls)]
        if class_name in target_classes:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 255
    return mask

def color_quantization(image, mask, k=8, soften=True):
    data = image.reshape((-1, 3))
    mask_flat = mask.flatten()
    non_essential_pixels = data[mask_flat == 0]

    kmeans = KMeans(n_clusters=k, random_state=42).fit(non_essential_pixels)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
    quantized_image = new_colors.reshape(image.shape).astype(np.uint8)

    if soften:
        quantized_image = cv2.bilateralFilter(quantized_image, d=9, sigmaColor=75, sigmaSpace=75)
    return quantized_image

def blend_images(original, quantized, mask, alpha=0.7):
    mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    blended = np.where(mask_expanded, original * (1 - alpha) + quantized * alpha, quantized)
    return blended.astype(np.uint8)

def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Processed image saved at: {output_path}")

def visualize_results(original, mask, final_image):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    ax[1].axis('off')

    ax[2].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Processed Image')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

def main(image_path, target_classes=['bird', 'tree'], k=8, alpha=0.7, output_dir="./output"):
    print(f"Processing image: {image_path}")
    image, original_size = load_image(image_path)
    model = load_yolo_model()
    mask = detect_objects(image, model, target_classes)
    quantized_image = color_quantization(image, mask, k)
    blended_image = blend_images(image, quantized_image, mask, alpha)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    os.makedirs(output_dir, exist_ok=True)
    save_image(blended_image, output_path)

    visualize_results(image, mask, blended_image)
    print(f"Finished processing image: {image_path}")

if __name__ == "__main__":
    main("/content/autistic test3.jpg", target_classes=['bird', 'tree'], k=30, alpha=0.7)
    main("/content/autistic test4.webp", target_classes=['bird', 'tree'], k=30, alpha=0.7)
    main("/content/autistic test5.jpg", target_classes=['bird', 'tree'], k=30, alpha=0.7)
