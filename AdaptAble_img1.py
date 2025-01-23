import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_image(image_path, target_size=(128, 128)):
    """Loads and resizes an image to the target size."""
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, target_size) / 255.0
    return image_resized, original_size

def build_unet(input_shape=(128, 128, 3)):
    """Build a simple U-Net using MobileNetV2 as the encoder."""
    inputs = tf.keras.Input(shape=input_shape)

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    encoder = base_model(inputs)

    x = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=2, padding='same')(encoder)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = tf.keras.Model(inputs, x)
    return model


def segment_image(model, image):
    """Generates a binary mask using the segmentation model."""
    pred_mask = model.predict(image[np.newaxis, ...])[0]
    mask = (pred_mask > 0.5).astype(np.uint8)

def color_quantization(image, mask, k=8, n_init=10, soften=True):
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    assert image.shape[:2] == mask.shape[:2], "Image and mask must have the same dimensions"

    data = image.reshape((-1, 3))
    mask_flat = mask.flatten()

    non_essential_pixels = data[mask_flat == 0]

    if non_essential_pixels.size == 0:
        raise ValueError("No non-essential pixels found for k-means clustering.")

    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42).fit(non_essential_pixels)

    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

    quantized_image = new_colors.reshape(image.shape)

    quantized_image = (quantized_image - quantized_image.min()) / (quantized_image.max() - quantized_image.min()) * 255
    quantized_image = quantized_image.astype(np.uint8)

    if soften:
        quantized_image = cv2.bilateralFilter(quantized_image, d=9, sigmaColor=75, sigmaSpace=75)

    return quantized_image

def visualize_results(original, mask, quantized_image):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    ax[1].axis('off')

    quantized_image = quantized_image.astype(np.uint8)
    ax[2].imshow(cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Quantized Image')
    ax[2].axis('off')

    plt.show()
import os
def main(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    image, original_size = load_image(image_path)
    model = build_unet()
    mask = segment_image(model, image)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    quantized_image = color_quantization(image, mask_resized, k=5)
    visualize_results(cv2.imread(image_path), mask_resized, quantized_image)

if __name__ == "__main__":
    main("/content/autistic test1.jpg")