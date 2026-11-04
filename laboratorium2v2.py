import csv

import numpy as np
import pyransac3d as pyrsc
from sklearn.cluster import DBSCAN, KMeans

INPUT_FILE = "chmury_polaczone.xyz"
RANSAC_THRESHOLD = 0.05
RANSAC_ITERATIONS = 1000
MAX_MEAN_DISTANCE = 0.1
MAX_TILT_DEGREES = 10


def read_xyz(file_path):
    with open(file_path, newline="") as file:
        reader = csv.reader(file, delimiter=" ")
        return np.array([[float(value) for value in row] for row in reader])


def plane_distances(points, normal, d):
    return np.abs(points @ normal + d)


def fit_plane_ransac(points, threshold=RANSAC_THRESHOLD, iterations=RANSAC_ITERATIONS, seed=None):
    rng = np.random.default_rng(seed)
    best_inliers = np.array([], dtype=int)

    for _ in range(iterations):
        p1, p2, p3 = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue

        normal = normal / norm
        d = -normal @ p1
        inliers = np.flatnonzero(plane_distances(points, normal, d) < threshold)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < 3:
        raise ValueError("Nie znaleziono płaszczyzny")

    centroid = points[best_inliers].mean(axis=0)
    _, _, vectors = np.linalg.svd(points[best_inliers] - centroid)
    normal = vectors[-1]
    return normal, -normal @ centroid, best_inliers


def describe_plane(points, normal, d):
    mean_distance = plane_distances(points, normal, d).mean()
    if mean_distance >= MAX_MEAN_DISTANCE:
        return mean_distance, "nie jest płaszczyzną"

    max_tilt = np.radians(MAX_TILT_DEGREES)
    vertical_component = abs(normal[2]) / np.linalg.norm(normal)
    if vertical_component >= np.cos(max_tilt):
        return mean_distance, "płaszczyzna pozioma"
    if vertical_component <= np.sin(max_tilt):
        return mean_distance, "płaszczyzna pionowa"
    return mean_distance, "płaszczyzna ukośna"


def format_vector(vector):
    largest = vector[np.argmax(np.abs(vector))]
    oriented = vector if largest >= 0 else -vector
    return "[" + ", ".join(f"{value + 0.0:.4f}" for value in np.round(oriented, 4)) + "]"


def print_result(index, points, normal, d, inliers):
    mean_distance, description = describe_plane(points, normal, d)
    print(f"Chmura {index}: {len(points)} punktów")
    print(f"  wektor normalny: {format_vector(normal)}")
    print(f"  punkty na płaszczyźnie: {len(inliers)}")
    print(f"  średnia odległość: {mean_distance:.4f}")
    print(f"  wynik: {description}")


def split_clouds(points, labels):
    return [points[labels == label] for label in np.unique(labels) if label != -1]


def run_kmeans_with_own_ransac(points):
    print("=== Zadania 3 i 4: k-średnich i własny RANSAC ===")
    labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(points)
    for index, cloud in enumerate(split_clouds(points, labels), start=1):
        normal, d, inliers = fit_plane_ransac(cloud, seed=0)
        print_result(index, cloud, normal, d, inliers)


def run_dbscan_with_pyransac3d(points):
    print("\n=== Zadanie 6: DBSCAN i pyransac3d ===")
    labels = DBSCAN(eps=1.0, min_samples=5).fit_predict(points)
    for index, cloud in enumerate(split_clouds(points, labels), start=1):
        equation, inliers = pyrsc.Plane().fit(
            cloud, thresh=RANSAC_THRESHOLD, maxIteration=RANSAC_ITERATIONS
        )
        normal = np.array(equation[:3])
        scale = np.linalg.norm(normal)
        print_result(index, cloud, normal / scale, equation[3] / scale, inliers)


if __name__ == "__main__":
    all_points = read_xyz(INPUT_FILE)
    print(f"Wczytano {len(all_points)} punktów\n")
    run_kmeans_with_own_ransac(all_points)
    run_dbscan_with_pyransac3d(all_points)