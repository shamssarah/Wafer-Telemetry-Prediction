
import math
import math
from turtle import width

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms.functional as F
from skimage.measure import block_reduce




def downsize_binary_preserve_foreground(img_array, block_h, block_w):
    """
    img_array: 2D numpy array, defect pixels = max value.
    block_h, block_w: how many source pixels map to each output pixel
                       (roughly source_size / target_size).
    Uses max-pooling so any defect pixel in a block survives.
    """
    return block_reduce(img_array, block_size=(block_h, block_w), func=np.max)

def resize_and_pad_image( img):  
    target_size = 32
    fill = 0  # Black padding

    width, height = img.size
    max_dim = max(width, height)
    if max_dim > target_size:
        block_h = max(1, math.ceil(height / target_size))
        block_w = max(1, math.ceil(width / target_size))
        img_arr = downsize_binary_preserve_foreground(np.array(img), block_h=block_h, block_w=block_w)
        img = F.to_pil_image(img_arr)
        width, height = img.size
        # fall through to padding below, don't return early
    else:
        scale = target_size / max_dim
        new_w, new_h = round(width * scale), round(height * scale)
        img = F.resize(img, (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)
        width, height = new_w, new_h
    # Pad the shorter dimension to reach target_size
    pad_w = target_size - width
    pad_h = target_size - height
    
    # Split the padding equally on both sides
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)
    return img


def get_defect_angle_and_linearity(img: np.ndarray, defect_value=None):
    """
    img: 2D numpy array (single-channel wafer map).
    defect_value: pixel value that represents a defect ("white").
        If None, assumes defect pixels are the maximum value in the image.

    Returns:
        angle_deg: dominant orientation angle in degrees, range [0, 180).
                   Returns None if fewer than 2 defect pixels are found
                   (can't fit a meaningful direction).
        linearity: float in [0, 1]. 1.0 = perfectly straight line,
                   0.0 = defect pixels spread equally in all directions.
                   Returns None if angle_deg is None.
    """
    if defect_value is None:
        defect_value = img.max()

    ys, xs = np.where(img == defect_value)
    if len(xs) < 2:
        return None, None

    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    coords -= coords.mean(axis=0)  # center at origin

    # Covariance matrix + eigen-decomposition (this IS the PCA step)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order

    # Largest eigenvalue / eigenvector = dominant direction
    major_idx = np.argmax(eigvals)
    minor_idx = 1 - major_idx
    major_vec = eigvecs[:, major_idx]

    # Angle of the major axis, in degrees, folded into [0, 180)
    angle_rad = np.arctan2(major_vec[1], major_vec[0])
    angle_deg = np.degrees(angle_rad) % 180

    # Linearity score: how dominant is the major axis vs the minor axis
    lam_major = eigvals[major_idx]
    lam_minor = eigvals[minor_idx]
    if lam_major + lam_minor == 0:
        linearity = 0.0
    else:
        linearity = float((lam_major - lam_minor) / (lam_major + lam_minor))

    return angle_deg, linearity


def analyze_scratch_orientations(images, defect_value=None, linearity_threshold=0.0):
    """
    images: list/array of 2D numpy arrays (Scratch-class wafer maps).
    linearity_threshold: only include samples with linearity >= this
        value in the angle histogram (use 0.0 to include everything).

    Returns:
        angles: list of angle_deg for samples that passed the threshold
        linearities: list of linearity scores for ALL samples
        skipped: count of samples with <2 defect pixels
    """
    angles = []
    linearities = []
    skipped = 0

    for img in images:
        img_pil = F.to_pil_image(img)
        img_resized_padded = resize_and_pad_image(img_pil)
        angle_deg, linearity = get_defect_angle_and_linearity(np.array(img_resized_padded), defect_value)
        if angle_deg is None:
            skipped += 1
            continue
        linearities.append(linearity)
        if linearity >= linearity_threshold:
            angles.append(angle_deg)

    return angles, linearities, skipped

def rank_by_linearity(images):
    """
    images: list of 2D numpy arrays.
 
    Returns a list of (index, angle_deg, linearity) tuples,
    sorted by linearity descending. Samples with <2 defect
    pixels are excluded.
    """
    results = []
    for i, img in enumerate(images):
        img_pil = F.to_pil_image(img)
        img_resized_padded = np.array(resize_and_pad_image(img_pil))
        angle_deg, linearity = get_defect_angle_and_linearity(img_resized_padded, defect_value=None)
        if linearity is not None:
            results.append((i, angle_deg, linearity, img.shape))
 
    results.sort(key=lambda r: r[2], reverse=True)
    return results


 
def plot_extremes(images, ranked_results, n=5):
    """
    Plots the top-n and bottom-n linearity samples.
    ranked_results: output of rank_by_linearity() (sorted descending).
    """
    top = ranked_results[:n]
    bottom = ranked_results[-n:]
 
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5))


    for col, (idx, angle, lin) in enumerate(top):
        img_pil = F.to_pil_image(images.iloc[idx])
        img_resized_padded = resize_and_pad_image(img_pil)
        print (idx,images.iloc[idx].shape, img_resized_padded.size)
        axes[0, col].imshow(img_resized_padded, cmap="gray")
        axes[0, col].set_title(f"idx={idx}\nlin={lin:.2f}, ang={angle:.0f}°", fontsize=9)
        axes[0, col].axis("off")
    axes[0, 0].set_ylabel("HIGH linearity", fontsize=11)
 
    for col, (idx, angle, lin) in enumerate(bottom):
        img_pil = F.to_pil_image(images.iloc[idx])
        img_resized_padded = resize_and_pad_image(img_pil)
        print (idx,images.iloc[idx].shape, img_resized_padded.size)
        axes[1, col].imshow(img_resized_padded, cmap="gray")
        axes[1, col].set_title(f"idx={idx}\nlin={lin:.2f}, ang={angle:.0f}°", fontsize=9)
        axes[1, col].axis("off")
    axes[1, 0].set_ylabel("LOW linearity", fontsize=11)
 
    plt.suptitle("Highest vs. lowest linearity Scratch samples")
    plt.tight_layout()
    plt.savefig("linearity_extremes.png", dpi=150)
    plt.show()
    print("Saved plot to linearity_extremes.png")
 


def plot_results(angles, linearities, bins=18):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Angle histogram (0-180 deg, since orientation is symmetric under 180 deg rotation)
    axes[0].hist(angles, bins=bins, range=(0, 180), color="steelblue", edgecolor="black")
    axes[0].set_title("Scratch orientation distribution")
    axes[0].set_xlabel("Angle (degrees)")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(range(0, 181, 30))

    # Linearity score distribution
    axes[1].hist(linearities, bins=20, range=(0, 1), color="darkorange", edgecolor="black")
    axes[1].set_title("Linearity score distribution\n(1.0 = straight line, 0.0 = scattered)")
    axes[1].set_xlabel("Linearity score")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("scratch_orientation_analysis.png", dpi=150)
    plt.show()
    print("Saved plot to scratch_orientation_analysis.png")

if __name__ == "__main__":
    train = pd.read_pickle('../data/original_raw/train_split.pkl')
    test = pd.read_pickle('../data/original_raw/test_data.pkl')
    validation = pd.read_pickle('../data/original_raw/val_data.pkl')

    scratch_images = train[train.failureType == "Scratch"].waferMap 
    scratch_images = pd.concat([scratch_images, test[test.failureType == "Scratch"].waferMap, validation[validation.failureType == "Scratch"].waferMap], axis=0)
    scratch_images = scratch_images.reset_index(drop=True)
    angles, linearities, skipped = analyze_scratch_orientations(
            scratch_images,
            defect_value=None,       # None = auto-detect max value as defect
            linearity_threshold=0.0  # set higher (e.g. 0.3) to filter out non-line-like samples
        )

    angles = np.array(angles)
    linearities = np.array(linearities)

    n = 10
    images = scratch_images[linearities >= 0.35]
    angles = np.array(angles)[linearities >= 0.35]
    training_samples = images[(angles >= 0) & (angles <= 90)]
    testing_samples = images[(angles > 90) & (angles <= 180)]

    pd.to_pickle(training_samples, "../data/original_raw/processed_scratch_training_samples.pkl")
    pd.to_pickle(testing_samples, "../data/original_raw/processed_scratch_testing_samples.pkl")

    print (f"Displaying {len(images)} Scratch samples with linearity >= 0.35")
    print (f"Training samples (0-90°): {len(training_samples)}")
    print (f"Testing samples (90-180°): {len(testing_samples)}")
    print ("skipped samples (too few defect pixels):", skipped)



# if __name__ == "__main__":
#     train = pd.read_pickle('../data/original_raw/train_split.pkl')
#     defect_samples = train[train.failureType == "Scratch"]
#     scratch_images = defect_samples.waferMap  # <-- fill this in


#     angles, linearities, skipped = analyze_scratch_orientations(
#         scratch_images,
#         defect_value=None,       # None = auto-detect max value as defect
#         linearity_threshold=0.0  # set higher (e.g. 0.3) to filter out non-line-like samples
#     )

#     print(f"Analyzed {len(angles)} samples, skipped {skipped} (too few defect pixels)")
#     print(f"Angle range: {min(angles):.1f} - {max(angles):.1f} deg" if angles else "No valid angles")
#     print(f"Mean linearity: {np.mean(linearities):.3f}" if linearities else "")

#     plot_results(angles, linearities)

#     ranked = rank_by_linearity(scratch_images)
#     print(f"Ranked {len(ranked)} samples by linearity")
#     print("Top 5:", [(i, f"{lin:.3f}, {shape}") for i, a, lin, shape in ranked[:5]])
#     print("Bottom 5:", [(i, f"{lin:.3f}, {shape}") for i, a, lin, shape in ranked[-5:]])
#     ranked = [[i,a,lin] for i,a,lin,shape in ranked]
#     plot_extremes(scratch_images, ranked, n=5)

