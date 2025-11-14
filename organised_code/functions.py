import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tifffile as tiff
from sklearn.cluster import KMeans
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches


def plot_image(image, title=""):
    # Plot the image
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap="gray")
    plt.colorbar(label="Pixel Value")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_example_reflectance(defect_cubes, sample_name, wavelength):
    # Get reflectance cube
    R = defect_cubes[sample_name]

    # Extract 2D image at selected wavelength
    wavelengths = list(range(450, 951, 20))
    reflectance_image = R[wavelengths.index(wavelength), :, :]
    title = f"Reflectance of {sample_name} at {wavelength} nm"

    plot_image(reflectance_image, title)


def plot_original_tif(tif_path):
    image = tiff.imread(tif_path)

    plot_image(image, tif_path)


def get_example_reflectance(defect_cubes, sample_name, wavelength):

    wavelengths = list(range(450, 951, 20))

    # Get reflectance cube
    R = defect_cubes[sample_name]

    # Extract 2D image at selected wavelength
    reflectance_image = R[wavelengths.index(wavelength), :, :]

    return reflectance_image


def gray_to_binary_image(gray):
    gray_norm = gray / np.max(gray)
    gray_norm = (gray_norm * 255).astype(np.uint8)
    _, binary_image = cv2.threshold(gray_norm, 70, 255, cv2.THRESH_BINARY_INV)

    return binary_image


def overlay_mask(base_image, mask, title, color=(1, 0, 0), alpha=0.5):
    base_image = base_image.astype(np.float32)
    if base_image.shape[0] == 3:
        base_image = np.transpose(base_image, (1, 2, 0))
    base_norm = base_image / 255.0 if base_image.max() > 1 else base_image

    overlay = np.zeros_like(base_norm)
    overlay[..., :3] = color

    blended = base_norm.copy()
    mask_bool = mask == 1
    blended[mask_bool] = (1 - alpha) * base_norm[mask_bool] + alpha * overlay[mask_bool]

    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(blended, 0, 1))
    plt.axis("off")
    plt.title(title)
    plt.show()


def compute_IC_mask(RGB_image, H=None, vis=True, pixel_IC=(192, 305)):
    RGB_image = RGB_image.astype(np.float32)

    if RGB_image.shape[0] == 3:
        RGB_image = np.transpose(RGB_image, (1, 2, 0))

    height, width, bands = RGB_image.shape
    pixels = RGB_image.reshape(-1, bands)
    pixels /= np.linalg.norm(pixels, axis=1, keepdims=True) + 1e-6

    k_means = KMeans(n_clusters=5, random_state=0, n_init=10)
    labels = k_means.fit_predict(pixels)
    segmented_image = labels.reshape(height, width)

    if H is not None:
        h, w, b = RGB_image.shape
        segmented_image = segmented_image.astype(np.float32)
        segmented_image = cv2.warpAffine(segmented_image, H, (w, h))

    # Plot clustering kmeans
    if vis:
        num_clusters = 5
        cmap_base = plt.get_cmap("tab10")
        colors = [cmap_base(i) for i in range(num_clusters)]
        cmap = ListedColormap(colors)
        plt.figure(figsize=(6, 4))
        im = plt.imshow(segmented_image, cmap=cmap)
        plt.title("K-Means Segmentation (5 Clusters)")
        plt.axis("off")
        patches = [
            mpatches.Patch(color=colors[i], label=f"Cluster {i+1}")
            for i in range(num_clusters)]
        plt.legend(
            handles=patches,
            loc="upper right",
            bbox_to_anchor=(1.25, 1),
            title="Clusters",)
        plt.show()

    # Get cluster that contains IC
    IC_label = segmented_image[(pixel_IC[1], pixel_IC[0])]
    mask = (segmented_image == IC_label).astype(np.uint8)

    # Using cluster region growing from center of IC
    flood_mask = mask.copy()
    cv2.floodFill(flood_mask, None, (pixel_IC[0], pixel_IC[1]), 128)
    circle_mask = (flood_mask == 128).astype(np.uint8)

    # Improve circle, fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(circle_mask, cv2.MORPH_CLOSE, kernel)
    filled = binary_fill_holes(closed > 0)

    if vis:
        plt.imshow(filled.astype(np.uint8))
        plt.title("Inital mask")
        plt.show()

    return filled.astype(np.uint8)


def align_and_visualise_homography(
    img_ref, img_to_align, defect, n_features=7000, visualise=True
):
    """    if img_ref.dtype != np.uint8:
            img_ref = np.clip(img_ref * 255, 0, 255).astype(np.uint8)
    if img_to_align.dtype != np.uint8:
            img_to_align = np.clip(img_to_align * 255, 0, 255).astype(np.uint8)"""

    img_to_align = cv2.equalizeHist(img_to_align)

    orb = cv2.ORB_create(n_features)
    kp1, des1 = orb.detectAndCompute(img_ref, None)
    kp2, des2 = orb.detectAndCompute(img_to_align, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_raw = bf.knnMatch(des1, des2, k=2)
    matches = [m for m, n in matches_raw if m.distance < 0.75 * n.distance]

    if len(matches) < 10:
        print(f"Not enough matches for {defect}")
        return None

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts_to = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Compute homography using RANSAC
    # H, mask = cv2.findHomography(
    #     pts_to, pts_ref, cv2.RANSAC, 5.0
    # )  # increase RANSAC threshold in case of noisy image

    # if H is None:
    #     print(f"Homography estimation failed for {defect}")
    #     return None

    # aligned = cv2.warpPerspective(img_to_align, H, (img_ref.shape[1], img_ref.shape[0]))

    M, mask = cv2.estimateAffinePartial2D(
        pts_to, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )

    if M is None:
        print(f"Similarity transform estimation failed for {defect}")
        return None

    aligned = cv2.warpAffine(img_to_align, M, (img_ref.shape[1], img_ref.shape[0]))

    # Visualization
    if visualise:
        matches_inliers = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
        match_vis = cv2.drawMatches(
            cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR),
            kp1,
            cv2.cvtColor(img_to_align, cv2.COLOR_GRAY2BGR),
            kp2,
            matches_inliers,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(match_vis)
        plt.title(f"Feature matches: {defect}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(aligned, cmap="gray")
        plt.title(f"Aligned binary ({defect})")
        plt.axis("off")

        overlay = cv2.addWeighted(
            cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR),
            0.5,
            cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR),
            0.5,
            0,
        )
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f"Overlap ({defect})")
        plt.axis("off")

        plt.show()

    return M


def align_and_blend_RGB_homography(ref_rgb, defect_rgb, H, defect_name):
    """
    Align defect_rgb to ref_rgb using the homography H (cv2.warpPerspective)
    """   
    if ref_rgb.dtype != np.uint8:
            ref_rgb = np.clip(ref_rgb * 255, 0, 255).astype(np.uint8)
    if defect_rgb.dtype != np.uint8:
            defect_rgb = np.clip(defect_rgb * 255, 0, 255).astype(np.uint8)
    h, w = ref_rgb.shape[:2]
    aligned_defect = cv2.warpAffine(defect_rgb, H, (w, h))
    blended = cv2.addWeighted(ref_rgb, 0.5, aligned_defect, 0.5, 0)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2RGB))
    plt.title("Reference RGB")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(aligned_defect, cv2.COLOR_BGR2RGB))
    plt.title(f"Aligned {defect_name}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title(f"Overlap ({defect_name})")
    plt.axis("off")

    plt.show()

    return aligned_defect, blended


def extract_RGB(cube, wavelengths):
    RGB = cube[
        [wavelengths.index(650), wavelengths.index(550), wavelengths.index(470)],
        :,
        :,
    ]
    RGB = np.transpose(RGB, (1, 2, 0))
    return RGB


def get_circle_info(mask, rgb_image, visualisation=False):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    centroid_x = M["m10"] / M["m00"]
    centroid_y = M["m01"] / M["m00"]

    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    center = (round(x), round(y))

    # Optional visualization
    if visualisation:
        output = rgb_image.copy()
        cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), -1)

        plt.imshow(output)
        plt.axis("off")
        plt.title("Final detected circle")
        plt.show()

    ellipse = cv2.fitEllipse(largest_contour) if len(largest_contour) >= 5 else None

    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

    return {
        "centroid": (centroid_x, centroid_y),
        "area": area,
        "perimeter": perimeter,
        "min_enclosing_circle": {"center": (x, y), "radius": radius},
        "ellipse": ellipse,
        "circularity": circularity,
        # "contour": largest_contour,
    }


def crop_circle(image, center, radius):
    x, y = center
    r = int(radius) # - 5

    x, y = int(round(x)), int(round(y))
    h, w = image.shape[:2]
    y1, y2 = max(0, y - r), min(h, y + r)
    x1, x2 = max(0, x - r), min(w, x + r)

    cropped_rect = image[y1:y2, x1:x2]

    mask = np.zeros(cropped_rect.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)

    if len(cropped_rect.shape) == 3:
        masked = cv2.bitwise_and(cropped_rect, cropped_rect, mask=mask)
    else:
        masked = cv2.bitwise_and(cropped_rect, cropped_rect, mask=mask)

    return masked


def average_reflectance_in_circle(hypercube, center, radius, transform=None):

    bands, h, w = hypercube.shape

    if transform is not None:
        if transform.shape == (3, 3):
            # Homography
            hypercube_aligned = np.stack(
                [cv2.warpPerspective(hypercube[i], transform, (w, h)) for i in range(bands)]
            )
        elif transform.shape == (2, 3):
            # Affine (similarity)
            hypercube_aligned = np.stack(
                [cv2.warpAffine(hypercube[i], transform, (w, h)) for i in range(bands)]
            )
        else:
            raise ValueError("Transform must be 2x3 (affine) or 3x3 (homography)")
    else:
        hypercube_aligned = hypercube

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(radius) - 5
    cv2.circle(mask, (cx, cy), r, 1, -1)

    mean_reflectance = []
    for i in range(bands):
        band = hypercube_aligned[i]
        masked_pixels = band[mask == 1]
        if masked_pixels.size > 0:
            mean_reflectance.append(np.mean(masked_pixels))
        else:
            mean_reflectance.append(np.nan)

    return np.array(mean_reflectance)


# def calculate_delta_E(LAB_ref, LAB_def, mask=None):
#     delta = LAB_ref - LAB_def
#     deltaE = np.sqrt(np.sum(delta**2, axis=2))
#     if mask is not None:
#         deltaE = deltaE[mask > 0]
#     return np.nanmean(deltaE)
import numpy as np


def calculate_delta_E(LAB_ref, LAB_def, mask=None):
    if mask is not None:
        LAB_ref = LAB_ref[mask > 0]
        LAB_def = LAB_def[mask > 0]

    L1, a1, b1 = LAB_ref[..., 0], LAB_ref[..., 1], LAB_ref[..., 2]
    L2, a2, b2 = LAB_def[..., 0], LAB_def[..., 1], LAB_def[..., 2]

    avg_L = 0.5 * (L1 + L2)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = 0.5 * (C1 + C2)

    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    avg_Lp = (L1 + L2) / 2
    avg_Cp = (C1p + C2p) / 2

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    avg_hp = np.where(
        (C1p * C2p) == 0,
        hsum / 2,
        np.where(
            hdiff <= 180,
            hsum / 2,
            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
        ),
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30))
        + 0.24 * np.cos(np.radians(2 * avg_hp))
        + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
        - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
    )

    delta_ro = 30 * np.exp(-(((avg_hp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
    Sl = 1 + (0.015 * (avg_Lp - 50) ** 2) / np.sqrt(20 + (avg_Lp - 50) ** 2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -np.sin(2 * np.radians(delta_ro)) * Rc

    deltaE00 = np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )

    return np.nanmean(deltaE00)


def interpolate_spectral_cube(
    spectral_cube, input_wavelengths, wl_min=300, wl_max=950, wl_step=1
):
    """Interpolate hyperspectral cube to a new wavelength range"""
    input_wavelengths = np.array(input_wavelengths)
    bands, H, W = spectral_cube.shape

    output_wavelengths = np.arange(wl_min, wl_max + wl_step, wl_step)
    cube_reshaped = spectral_cube.reshape(bands, -1)

    interp_func = interp1d(
        input_wavelengths,
        cube_reshaped,
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )

    cube_interp = interp_func(output_wavelengths)
    spectral_cube_interp = cube_interp.reshape(len(output_wavelengths), H, W)

    return spectral_cube_interp, output_wavelengths

def preprocess(img):
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # Smooth edges to reduce noise
    img = cv2.GaussianBlur(img, (7,7), 1.5)
    # Optional: sharpen edges
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def detect_circle_center(image, vis=False):
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = preprocess(image)

    circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=80,
    param1=80,   # lower Canny threshold
    param2=20,   # LOWER = more sensitive
    minRadius=20,
    maxRadius=80
    )

    # Draw only the first detected circle
    output = image.copy()
    center = None 
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Circle outline
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Center point
        # Extract center coordinates
        center = (int(np.round(x)), int(np.round(y)))

    if vis:
        plt.imshow(output)
        plt.title("Initial circle and center detection")
        plt.show()

    return center