import numpy as np


def _check_positive(**values):
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"Parametr {name} musi być dodatni, otrzymano: {value}")


def generate_horizontal_plane(width, length, num_points):
    _check_positive(width=width, length=length, num_points=num_points)

    x = np.random.uniform(-width / 2, width / 2, num_points)
    y = np.random.uniform(-length / 2, length / 2, num_points)
    z = np.zeros(num_points)

    return np.column_stack((x, y, z))


def generate_vertical_plane(width, height, num_points):
    _check_positive(width=width, height=height, num_points=num_points)

    x = np.random.uniform(-width / 2, width / 2, num_points)
    y = np.zeros(num_points)
    z = np.random.uniform(0, height, num_points)

    return np.column_stack((x, y, z))


def generate_cylinder(radius, height, num_points):
    _check_positive(radius=radius, height=height, num_points=num_points)

    angle = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = np.random.uniform(0, height, num_points)

    return np.column_stack((x, y, z))


if __name__ == "__main__":
    clouds = {
        "Płaszczyzna pozioma": generate_horizontal_plane(10, 20, 2000),
        "Płaszczyzna pionowa": generate_vertical_plane(10, 5, 2000),
        "Walec": generate_cylinder(3, 8, 2000),
    }

    for name, points in clouds.items():
        print(name)
        print(f"  liczba punktów: {len(points)}")
        print(f"  minimum (x, y, z): {points.min(axis=0).round(2)}")
        print(f"  maksimum (x, y, z): {points.max(axis=0).round(2)}")