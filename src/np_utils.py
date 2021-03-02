import cv2
import numpy as np


def normalize_img(img):
    return img / 127.5 - 1


def squared_euclidean_distance_np(a, b):
    b = b.T
    a2 = np.sum(np.square(a), axis=1)
    b2 = np.sum(np.square(b), axis=0)
    ab = np.matmul(a, b)
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d


def color_quantize_np(x, clusters):
    x = normalize_img(x)
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_np(x, clusters)
    return np.argmin(d, axis=1)


def generate_primer(
    path_to_image, samples, gpus, size, color_clusters_dir, n_px_crop=16
):
    image_path = (
        path_to_image  # Change this you you already have an image on your machine
    )
    batch = [image_path] * (gpus * samples)

    dim = (size, size)

    x = np.zeros((gpus * samples, size, size, 3), dtype=np.uint8)

    for n, image_path in enumerate(batch):

        img_np = cv2.imread(image_path)
        img_np = cv2.cvtColor(
            img_np, cv2.COLOR_BGR2RGB
        )  # BGR -> RGB, default one is BGR and we need RGB
        H, W, C = img_np.shape
        D = min(H, W)
        img_np = img_np[:D, :D, :C]  # crop square image with shorter dim
        x[n] = cv2.resize(img_np, dim, interpolation=cv2.INTER_AREA)

    clusters = np.load(f"{color_clusters_dir}")
    samples = color_quantize_np(x, clusters).reshape(x.shape[:-1])
    primers = samples.reshape(-1, size * size)[
        :, : n_px_crop * size
    ]  # crop top n_px_crop rows
    return primers
