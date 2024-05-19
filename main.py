import cv2
import numpy as np
from matplotlib import pyplot as plt


def resize(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def build_gaussian_pyramid(image):
    pyramid = [image]
    for _ in range(3):  # Adjust the number of levels as needed
        image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian blur
        image = resize(image, 0.5)
        pyramid.append(image)
    return pyramid

def build_laplacian_pyramid(image):
    pyramid = [image]
    for _ in range(3):  # Adjust the number of levels as needed
        if image.size == 0:
            break
        image = resize(image, 0.5)
        pyramid.append(image)

    laplacian_pyramid = [pyramid[-1].astype(np.float32)]  # Convert to float to retain color information
    for i in range(len(pyramid) - 1, 0, -1):
        expanded = cv2.resize(pyramid[i], (pyramid[i-1].shape[1], pyramid[i-1].shape[0]), interpolation=cv2.INTER_AREA)
        kernel = np.ones((5, 5), np.float32) / 25
        expanded = cv2.filter2D(expanded, -1, kernel)
        laplacian = pyramid[i - 1].astype(np.float32) - expanded
        laplacian_pyramid.insert(0, laplacian)

    return laplacian_pyramid

def blend_pyramids(laplacian1, laplacian2, mask_pyramid):
    blended_pyramid = []

    num_levels = min(len(laplacian1), len(laplacian2), len(mask_pyramid))

    for i in range(num_levels):
        mask = mask_pyramid[i]

        # Resize the Laplacian levels to match the mask dimensions using INTER_LINEAR
        laplacian1_resized = cv2.resize(laplacian1[i], (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        laplacian2_resized = cv2.resize(laplacian2[i], (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Ensure both Laplacian levels and mask are in float32
        laplacian1_resized = laplacian1_resized.astype(np.float32)
        laplacian2_resized = laplacian2_resized.astype(np.float32)
        mask = mask.astype(np.float32)

        # If the Laplacian levels are color images, convert the mask to the same number of channels
        if len(laplacian1_resized.shape) == 3 and mask.shape[-1] == 1:
            mask = np.repeat(mask, 3, axis=-1)

        # Perform color blending for each channel
        blended_channel = (laplacian1_resized * mask) + (laplacian2_resized * (1 - mask))
        blended_pyramid.append(blended_channel)

    return blended_pyramid







def collapse_pyramid(pyramid):
    # Start with the smallest level
    image = pyramid[0].astype(np.float32)

    for i in range(1, len(pyramid)):
        # Resize the current level to the size of the previous level
        level_resized = cv2.resize(pyramid[i], (image.shape[1], image.shape[0]))

        # Add the expanded image with the current level
        image += level_resized

    return np.clip(image, 0, 255).astype(np.uint8)  # Clip values to valid range

# Load your images
image1 = cv2.imread("path here")
image2 = cv2.imread("path here")

# Assuming you have one region mask (replace the file path with your actual mask image path)
cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
roi = cv2.selectROI("Select ROI", image1)
cv2.destroyWindow("Select ROI")  # Close the temporary window

mask = np.zeros_like(image1)
mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 1



# Build Gaussian pyramids for the images
gaussian_pyramid1 = build_gaussian_pyramid(image1)
gaussian_pyramid2 = build_gaussian_pyramid(image2)

# Build Laplacian pyramids for the images
laplacian_pyramid1 = build_laplacian_pyramid(image1)  # Use the last level of the Gaussian pyramid
laplacian_pyramid2 = build_laplacian_pyramid(image2)


# Build Gaussian pyramid for the mask
gaussian_pyramid_mask = build_gaussian_pyramid(mask)

# Blend Laplacian pyramids using the region mask
blended_pyramid = blend_pyramids(laplacian_pyramid1, laplacian_pyramid2, gaussian_pyramid_mask)

# Collapse the blended Laplacian pyramid to get the final blended image
final_image = collapse_pyramid(blended_pyramid)




# Function to save an image with a specific filename and path
def save_image(image, filename_prefix, level, output_path):
    filename = fr"{output_path}\{filename_prefix}_level_{level}.png"
    cv2.imwrite(filename, image)
    print(f"Saved: {filename}")

# Specify the output path
output_path = r"C:\Users\Sara\Desktop\imageee"

# Save Gaussian pyramids for both images
for i, img in enumerate(gaussian_pyramid1):
    save_image(img, 'gaussian_pyramid1', i, output_path)

for i, img in enumerate(gaussian_pyramid2):
    save_image(img, 'gaussian_pyramid2', i, output_path)

# Save Laplacian pyramids for both images
for i, img in enumerate(laplacian_pyramid1):
    save_image(cv2.convertScaleAbs(img), 'laplacian_pyramid1', i, output_path)

for i, img in enumerate(laplacian_pyramid2):
    save_image(cv2.convertScaleAbs(img), 'laplacian_pyramid2', i, output_path)

# Save Gaussian pyramid for the mask
for i, img in enumerate(gaussian_pyramid_mask):
    save_image(img, 'gaussian_pyramid_mask', i, output_path)

# Save Blended Laplacian pyramid
for i, img in enumerate(blended_pyramid):
    save_image(cv2.convertScaleAbs(img), 'blended_pyramid', i, output_path)



# Save the final blended image
final_output_path = fr"{output_path}\output_final.jpg"
cv2.imwrite(final_output_path, final_image)
print(f"Saved: {final_output_path}")

# Display the final blended image
cv2.imshow("Final Blended Image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
