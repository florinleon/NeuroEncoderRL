"""
Multi-dimensional topology tests for NeuralSheetEncoder.

This script computes similarity between the code for the origin and:
  1) A 1D line x' in [0,1] (x varies, other coordinates stay at 0).
  2) The vertices of the 2D unit square plus its center.
  3) The vertices of the 3D unit cube plus its center.
  4) The vertices of the 4D unit hypercube plus its center.

For each test, it first sets settings.InputDimensionCount to the
corresponding dimensionality (1, 2, 3, or 4), then reloads the
encoder module so that the encoder is constructed with that input size.
"""

import math
import random
import importlib
import settings
import neural_sheet_encoder


def _configure_input_dimension(d):
    """Set input dimensionality in settings and reload the encoder module.

    This changes settings.InputDimensionCount, then reloads the encoder so that
    any new NeuralSheetEncoder instance is constructed with the updated input
    dimensionality.
    """
    settings.InputDimensionCount = d
    importlib.reload(neural_sheet_encoder)


def _generate_hypercube_vertices(dim):
    """
    Generate all vertices of the {0,1}^dim hypercube except the origin.
    """
    # Each mask from 1 to 2**dim - 1 encodes one vertex by treating its bits
    # as on/off switches for the coordinates.
    vertices = []
    for mask in range(1, 2 ** dim):  # skip mask == 0 (origin)
        coords = [(1.0 if (mask >> bit) & 1 else 0.0) for bit in range(dim)]
        vertices.append(coords)
    return vertices


def run_one_dimensional_line_average(runs, steps=10, master_seed=None):
    """
    Similarity between origin and points on the 1D line x' in [0,1].
    Only the first coordinate varies.
    """

    settings.SparsityPercentile = 0.98
    settings.GaussianSigma = 1
    settings.DetectorsPerDimension = 100

    _configure_input_dimension(1)

    global_rng = random.Random(master_seed)
    sum_similarities = [0.0 for _ in range(steps + 1)]

    for _ in range(runs):
        encoder_seed = global_rng.randint(0, 2**31 - 1)
        encoder = neural_sheet_encoder.NeuralSheetEncoder(random_seed=encoder_seed)

        origin = [0.0]
        origin_code = encoder.encode(origin)

        for i in range(steps + 1):
            x = i / steps
            vec = [x]
            code = encoder.encode(vec)
            similarity = encoder.similarity(origin_code, code)
            sum_similarities[i] += similarity

    print(f"1D line test x' in [0,1] (average over {runs} runs)")
    print("x'\tDistance\tAvgSimilarity\n")

    for i in range(steps + 1):
        x = i / steps
        distance = abs(x - 0.0)
        avg_similarity = sum_similarities[i] / runs
        print(f"{x:.2f}\t{distance:.3f}\t\t{avg_similarity:.3f}")

    # 1D change in 3D space
    # The settings below deliberately differ from the 1D case so that the
    # encoder can be probed in a denser, lower-sparsity regime in 3D.

    settings.SparsityPercentile = 0.95
    settings.GaussianSigma = 0.5
    settings.DetectorsPerDimension = 10

    _configure_input_dimension(3)

    global_rng = random.Random(master_seed)
    sum_similarities = [0.0 for _ in range(steps + 1)]

    for _ in range(runs):
        encoder_seed = global_rng.randint(0, 2**31 - 1)
        encoder = neural_sheet_encoder.NeuralSheetEncoder(random_seed=encoder_seed)

        origin = [0.0, 0.0, 0.0]
        origin_code = encoder.encode(origin)

        for i in range(steps + 1):
            x = i / steps
            vec = [x, 0, 0]
            code = encoder.encode(vec)
            similarity = encoder.similarity(origin_code, code)
            sum_similarities[i] += similarity

    print(f"1D line test x' in [0,1]^3 (average over {runs} runs)")
    print("x'\tDistance\tAvgSimilarity\n")

    for i in range(steps + 1):
        x = i / steps
        distance = abs(x - 0.0)
        avg_similarity = sum_similarities[i] / runs
        print(f"{x:.2f}\t{distance:.3f}\t\t{avg_similarity:.3f}")


def run_two_dimensional_square_vertices_and_center_average(runs, master_seed=None):
    """
    Similarity between (0,0) and the three other vertices of the unit square
    plus the square center (0.5,0.5).
    """
    _configure_input_dimension(2)

    # Logical 2D coordinates
    points_2d = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],  # center
    ]

    # Euclidean distances in 2D from the origin
    distances = [math.sqrt(x * x + y * y) for (x, y) in points_2d]

    global_rng = random.Random(master_seed)
    sum_similarities = [0.0 for _ in points_2d]

    for _ in range(runs):
        encoder_seed = global_rng.randint(0, 2**31 - 1)
        encoder = neural_sheet_encoder.NeuralSheetEncoder(random_seed=encoder_seed)

        origin = [0.0, 0.0]
        origin_code = encoder.encode(origin)

        for idx, p in enumerate(points_2d):
            code = encoder.encode(p)
            similarity = encoder.similarity(origin_code, code)
            sum_similarities[idx] += similarity

    print(f"2D square vertices and center (average over {runs} runs)")
    print("Point\t\tDistance\tAvgSimilarity\n")

    for (x, y), dist, sum_sim in zip(points_2d, distances, sum_similarities):
        avg_similarity = sum_sim / runs
        print(f"({x:.1f},{y:.1f})\t{dist:.3f}\t\t{avg_similarity:.3f}")


def run_three_dimensional_cube_vertices_and_center_average(runs, master_seed=None):
    """
    Similarity between (0,0,0) and the seven other vertices of the unit cube
    plus the cube center (0.5,0.5,0.5).
    """
    _configure_input_dimension(3)

    vertices_3d = _generate_hypercube_vertices(3)
    center_3d = [0.5, 0.5, 0.5]
    points_3d = vertices_3d + [center_3d]

    distances = [math.sqrt(x * x + y * y + z * z) for (x, y, z) in points_3d]

    global_rng = random.Random(master_seed)
    sum_similarities = [0.0 for _ in points_3d]

    for _ in range(runs):
        encoder_seed = global_rng.randint(0, 2**31 - 1)
        encoder = neural_sheet_encoder.NeuralSheetEncoder(random_seed=encoder_seed)

        origin = [0.0, 0.0, 0.0]
        origin_code = encoder.encode(origin)

        for idx, p in enumerate(points_3d):
            code = encoder.encode(p)
            similarity = encoder.similarity(origin_code, code)
            sum_similarities[idx] += similarity

    print(f"3D cube vertices and center (average over {runs} runs)")
    print("Point\t\t\tDistance\tAvgSimilarity\n")

    for (x, y, z), dist, sum_sim in zip(points_3d, distances, sum_similarities):
        avg_similarity = sum_sim / runs
        print(
            f"({x:.1f},{y:.1f},{z:.1f})\t"
            f"{dist:.3f}\t\t{avg_similarity:.3f}"
        )


def run_four_dimensional_hypercube_vertices_and_center_average(runs, master_seed=None):
    """
    Similarity between (0,0,0,0) and the 15 other vertices of the 4D unit
    hypercube plus the hypercube center (0.5,0.5,0.5,0.5).
    """
    _configure_input_dimension(4)

    vertices_4d = _generate_hypercube_vertices(4)
    center_4d = [0.5, 0.5, 0.5, 0.5]
    points_4d = vertices_4d + [center_4d]

    distances = []
    for w, x, y, z in points_4d:
        distances.append(math.sqrt(w * w + x * x + y * y + z * z))

    global_rng = random.Random(master_seed)
    sum_similarities = [0.0 for _ in points_4d]

    for _ in range(runs):
        encoder_seed = global_rng.randint(0, 2**31 - 1)
        encoder = neural_sheet_encoder.NeuralSheetEncoder(random_seed=encoder_seed)

        origin = [0.0, 0.0, 0.0, 0.0]
        origin_code = encoder.encode(origin)

        for idx, p in enumerate(points_4d):
            code = encoder.encode(p)
            similarity = encoder.similarity(origin_code, code)
            sum_similarities[idx] += similarity

    print(f"4D hypercube vertices and center (average over {runs} runs)")
    print("Point\t\t\t\tDistance\tAvgSimilarity\n")

    for (w, x, y, z), dist, sum_sim in zip(points_4d, distances, sum_similarities):
        avg_similarity = sum_sim / runs
        print(f"({w:.1f},{x:.1f},{y:.1f},{z:.1f})\t{dist:.3f}\t\t{avg_similarity:.3f}")


if __name__ == "__main__":
    runs = 100

    run_one_dimensional_line_average(runs)
    print()
    run_two_dimensional_square_vertices_and_center_average(runs)
    print()
    run_three_dimensional_cube_vertices_and_center_average(runs)
    print()
    run_four_dimensional_hypercube_vertices_and_center_average(runs)
