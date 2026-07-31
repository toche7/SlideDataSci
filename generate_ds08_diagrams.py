"""Generate explanatory diagrams for the DS08 dimensionality-reduction deck."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import umap
from sklearn.datasets import load_breast_cancer, load_digits, make_blobs
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection


OUTPUT_DIR = Path("fig/diagrams")
RNG = np.random.default_rng(42)
COLORS = ["#2563eb", "#d97706", "#059669"]


def save_figure(figure: plt.Figure, filename: str, *, w_pad=None) -> None:
    figure.tight_layout(w_pad=w_pad)
    figure.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_pca_projection() -> None:
    """Show perpendicular projection to the first principal component."""
    points = RNG.multivariate_normal([0, 0], [[5.0, 3.6], [3.6, 3.2]], size=70)
    _, _, vectors = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    direction = vectors[0]
    projections = np.outer((points - points.mean(axis=0)) @ direction, direction) + points.mean(axis=0)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axis = axes[0]
    axis.scatter(points[:, 0], points[:, 1], s=38, color="#2563eb", alpha=0.8)
    for point, projection in zip(points[::4], projections[::4]):
        axis.plot([point[0], projection[0]], [point[1], projection[1]], color="#94a3b8", lw=1)
    line = np.vstack([points.mean(axis=0) - 3.2 * direction, points.mean(axis=0) + 3.2 * direction])
    axis.plot(line[:, 0], line[:, 1], color="#dc2626", lw=3, label="PC1")
    axis.set_title("Original space: project onto PC1", weight="bold")
    axis.set_aspect("equal")
    axis.legend(frameon=False)
    axis.axis("off")

    axis = axes[1]
    coordinate = (points - points.mean(axis=0)) @ direction
    axis.scatter(coordinate, np.zeros_like(coordinate), s=38, color="#059669", alpha=0.8)
    axis.axhline(0, color="#475569", lw=1.5)
    axis.set_title("Compressed representation: one coordinate z", weight="bold")
    axis.set_xlabel("z = U_reduce^T x")
    axis.set_yticks([])
    axis.spines[["left", "right", "top"]].set_visible(False)
    save_figure(figure, "08_pca_projection.png")


def create_pca_3d_to_2d() -> None:
    """Show a 3D point cloud being projected onto its first two PCA components."""
    latent = RNG.normal(size=(160, 2))
    transformation = np.array([[1.2, 0.3], [0.4, 1.0], [0.8, -0.5]])
    data = latent @ transformation.T + RNG.normal(scale=0.14, size=(len(latent), 3))
    pca = PCA(n_components=2).fit(data)
    embedded = pca.transform(data)
    center = pca.mean_
    basis = pca.components_

    figure = plt.figure(figsize=(10, 4.4))
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_3d.scatter(data[:, 0], data[:, 1], data[:, 2], c=embedded[:, 0], cmap="viridis", s=22, alpha=0.82)
    grid = np.linspace(-3, 3, 9)
    grid_x, grid_y = np.meshgrid(grid, grid)
    plane = center[:, None, None] + basis[0, :, None, None] * grid_x + basis[1, :, None, None] * grid_y
    axis_3d.plot_surface(plane[0], plane[1], plane[2], color="#60a5fa", alpha=0.28, linewidth=0)
    axis_3d.set_title("Original data in 3D", weight="bold", pad=12)
    axis_3d.set_xlabel("x1")
    axis_3d.set_ylabel("x2")
    axis_3d.set_zlabel("x3")
    axis_3d.view_init(elev=25, azim=-50)

    axis_2d = figure.add_subplot(1, 2, 2)
    axis_2d.scatter(embedded[:, 0], embedded[:, 1], c=embedded[:, 0], cmap="viridis", s=28, alpha=0.82)
    axis_2d.axhline(0, color="#cbd5e1", lw=1)
    axis_2d.axvline(0, color="#cbd5e1", lw=1)
    axis_2d.set_title("Projected onto PC1 and PC2", weight="bold")
    axis_2d.set_xlabel("z1 = PC1")
    axis_2d.set_ylabel("z2 = PC2")
    axis_2d.set_aspect("equal")
    axis_2d.grid(alpha=0.18)
    save_figure(figure, "08_pca_3d_to_2d.png", w_pad=5.0)


def create_pca_scree_plot() -> None:
    """Show individual and cumulative explained variance for the PCA workshop dataset."""
    features, _ = load_breast_cancer(return_X_y=True)
    explained_variance = PCA().fit(StandardScaler().fit_transform(features)).explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    components = np.arange(1, len(explained_variance) + 1)
    component_90 = np.searchsorted(cumulative_variance, 0.90) + 1
    component_95 = np.searchsorted(cumulative_variance, 0.95) + 1

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(components, explained_variance, color="#2563eb", width=0.8)
    axes[0].set(
        title="Scree plot: variance per component",
        xlabel="Principal component",
        ylabel="Explained variance ratio",
        xlim=(0.5, len(components) + 0.5),
    )

    axes[1].plot(components, cumulative_variance, color="#059669", marker="o", markersize=3, lw=2)
    axes[1].axhline(0.90, color="#d97706", linestyle="--", label="90% target")
    axes[1].axhline(0.95, color="#dc2626", linestyle="--", label="95% target")
    axes[1].axvline(component_90, color="#d97706", alpha=0.45, linestyle=":")
    axes[1].axvline(component_95, color="#dc2626", alpha=0.45, linestyle=":")
    axes[1].annotate(f"{component_90} PCs", (component_90, 0.90), xytext=(5, -25), textcoords="offset points", color="#b45309")
    axes[1].annotate(f"{component_95} PCs", (component_95, 0.95), xytext=(5, 8), textcoords="offset points", color="#b91c1c")
    axes[1].set(
        title="Cumulative explained variance",
        xlabel="Number of components retained",
        ylabel="Cumulative variance ratio",
        xlim=(0.5, len(components) + 0.5),
        ylim=(0, 1.04),
    )
    axes[1].legend(frameon=False, loc="lower right")
    for axis in axes:
        axis.grid(alpha=0.18)
    save_figure(figure, "08_pca_scree_plot.png")


def create_pca_reconstruction_error() -> None:
    """Compare low- and high-rank PCA reconstructions against the original data."""
    points = RNG.multivariate_normal([0, 0], [[4.0, 2.5], [2.5, 2.2]], size=75)
    low_rank = PCA(n_components=1).fit(points)
    high_rank = PCA(n_components=2).fit(points)
    low_reconstruction = low_rank.inverse_transform(low_rank.transform(points))
    high_reconstruction = high_rank.inverse_transform(high_rank.transform(points))
    low_error = np.mean(np.sum((points - low_reconstruction) ** 2, axis=1))
    high_error = np.mean(np.sum((points - high_reconstruction) ** 2, axis=1))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, reconstruction, components, error in [
        (axes[0], low_reconstruction, 1, low_error),
        (axes[1], high_reconstruction, 2, high_error),
    ]:
        axis.scatter(points[:, 0], points[:, 1], color="#94a3b8", s=35, alpha=0.75, label="Original x")
        axis.scatter(reconstruction[:, 0], reconstruction[:, 1], color="#2563eb", s=28, alpha=0.85, label="Reconstructed x-hat")
        for original, recovered in zip(points[::5], reconstruction[::5]):
            axis.plot([original[0], recovered[0]], [original[1], recovered[1]], color="#dc2626", lw=1.2, alpha=0.75)
        axis.set_title(f"k = {components}: mean squared error = {error:.2f}", weight="bold")
        axis.set_aspect("equal")
        axis.grid(alpha=0.18)
        axis.set_xlabel("Feature 1")
        axis.set_ylabel("Feature 2")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    save_figure(figure, "08_pca_reconstruction_error.png")


def create_pca_project_reconstruct_flow() -> None:
    """Illustrate x -> z -> x-hat for a single PCA projection and reconstruction."""
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 0.7, 1.25]})
    original, reduced, reconstructed = axes

    line_x = np.array([-2.5, 2.5])
    line_y = 0.65 * line_x
    point = np.array([1.45, 0.25])
    direction = np.array([1.0, 0.65])
    direction /= np.linalg.norm(direction)
    projected = direction * (point @ direction)

    for axis, title in [(original, "Original space"), (reconstructed, "Reconstructed space")]:
        axis.axhline(0, color="#94a3b8", lw=1)
        axis.axvline(0, color="#94a3b8", lw=1)
        axis.plot(line_x, line_y, color="#059669", lw=2.5, label="PC1 direction")
        axis.set(xlim=(-2.8, 2.8), ylim=(-2.2, 2.2), title=title)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    original.scatter(*point, s=90, color="#dc2626", zorder=3, label="x")
    original.scatter(*projected, s=70, color="#2563eb", zorder=3, label="projection")
    original.plot([point[0], projected[0]], [point[1], projected[1]], color="#dc2626", lw=2, linestyle="--")
    original.annotate("x", point, xytext=(8, 8), textcoords="offset points", color="#b91c1c", weight="bold")
    original.legend(frameon=False, fontsize=8, loc="upper left")

    reduced.axhline(0, color="#475569", lw=2)
    coordinate = point @ direction
    reduced.scatter(coordinate, 0, s=90, color="#2563eb", zorder=3)
    reduced.annotate(r"$z$", (coordinate, 0), xytext=(0, 14), textcoords="offset points", ha="center", color="#1d4ed8", weight="bold")
    reduced.set(xlim=(-2.8, 2.8), ylim=(-0.8, 0.8), title="Reduced space")
    reduced.set_xlabel(r"$z = U_{\mathrm{reduce}}^T x$")
    reduced.set_yticks([])
    for spine in reduced.spines.values():
        spine.set_visible(False)

    reconstructed.scatter(*projected, s=90, color="#2563eb", zorder=3, label="x-hat")
    reconstructed.annotate(r"$\hat{x}$", projected, xytext=(8, -16), textcoords="offset points", color="#1d4ed8", weight="bold")
    reconstructed.legend(frameon=False, fontsize=8, loc="upper left")

    figure.add_artist(FancyArrowPatch((0.365, 0.52), (0.43, 0.52), transform=figure.transFigure, arrowstyle="->", mutation_scale=18, color="#334155", lw=2))
    figure.add_artist(FancyArrowPatch((0.57, 0.52), (0.635, 0.52), transform=figure.transFigure, arrowstyle="->", mutation_scale=18, color="#334155", lw=2))
    figure.text(0.397, 0.57, "project", ha="center", color="#334155", weight="bold")
    figure.text(0.602, 0.57, "reconstruct", ha="center", color="#334155", weight="bold")
    save_figure(figure, "08_pca_project_reconstruct_flow.png", w_pad=5.5)


def create_lda_separation() -> None:
    """Compare a noisy 2D class view with a discriminant one-dimensional projection."""
    class_a = RNG.multivariate_normal([-1.1, 0.1], [[1.5, 0.9], [0.9, 1.0]], size=45)
    class_b = RNG.multivariate_normal([1.1, -0.1], [[1.5, 0.9], [0.9, 1.0]], size=45)
    data = np.vstack([class_a, class_b])
    labels = np.array([0] * len(class_a) + [1] * len(class_b))
    within = np.cov(class_a, rowvar=False) + np.cov(class_b, rowvar=False)
    direction = np.linalg.solve(within, class_b.mean(axis=0) - class_a.mean(axis=0))
    direction /= np.linalg.norm(direction)
    projected = data @ direction

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label, color, name in [(0, COLORS[0], "Class A"), (1, COLORS[1], "Class B")]:
        axes[0].scatter(data[labels == label, 0], data[labels == label, 1], color=color, s=38, alpha=0.8, label=name)
        axes[1].scatter(projected[labels == label], np.full((labels == label).sum(), label), color=color, s=38, alpha=0.8)
    origin = data.mean(axis=0)
    axes[0].arrow(origin[0], origin[1], 1.6 * direction[0], 1.6 * direction[1], width=0.025, color="#dc2626")
    axes[0].set_title("Original features", weight="bold")
    axes[0].legend(frameon=False)
    axes[0].set_aspect("equal")
    axes[1].set_title("LDA projection: clearer class separation", weight="bold")
    axes[1].set_xlabel("w^T x")
    axes[1].set_yticks([0, 1], ["Class A", "Class B"])
    for axis in axes:
        axis.grid(alpha=0.18)
    save_figure(figure, "08_lda_class_separation.png")


def create_random_projection() -> None:
    """Compare pairwise distances before and after a random projection."""
    high_dimensional, _ = make_blobs(n_samples=100, centers=3, n_features=24, cluster_std=3.0, random_state=42)
    reduced = GaussianRandomProjection(n_components=7, random_state=42).fit_transform(high_dimensional)
    pairs = RNG.integers(0, len(high_dimensional), size=(350, 2))
    original_distance = np.linalg.norm(high_dimensional[pairs[:, 0]] - high_dimensional[pairs[:, 1]], axis=1)
    projected_distance = np.linalg.norm(reduced[pairs[:, 0]] - reduced[pairs[:, 1]], axis=1)
    nonzero_distance = original_distance > 0
    original_distance = original_distance[nonzero_distance]
    projected_distance = projected_distance[nonzero_distance]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(original_distance, projected_distance, color="#2563eb", s=22, alpha=0.65)
    bound = max(original_distance.max(), projected_distance.max())
    axes[0].plot([0, bound], [0, bound], "--", color="#dc2626", label="perfect preservation")
    axes[0].set(xlabel="Distance in 24D", ylabel="Distance after RP", title="Pairwise distances remain similar")
    axes[0].legend(frameon=False, fontsize=8)

    distortion = np.abs(projected_distance - original_distance) / original_distance
    axes[1].hist(distortion, bins=20, color="#059669", edgecolor="white")
    axes[1].axvline(distortion.mean(), color="#dc2626", lw=2, label=f"mean = {distortion.mean():.2f}")
    axes[1].set(xlabel="Relative distance distortion", ylabel="Pair count", title="Distortion distribution")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    save_figure(figure, "08_random_projection_distances.png")


def create_umap_neighborhoods() -> None:
    """Illustrate local neighborhoods and weighted graph edges used by UMAP."""
    left = RNG.multivariate_normal([-1.1, 0.1], [[0.25, 0.05], [0.05, 0.18]], size=22)
    right = RNG.multivariate_normal([1.0, 0.5], [[0.35, -0.08], [-0.08, 0.22]], size=22)
    points = np.vstack([left, right])
    neighbors = NearestNeighbors(n_neighbors=5).fit(points).kneighbors(return_distance=False)
    focus = 12

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(points[:, 0], points[:, 1], s=42, color="#94a3b8")
    axes[0].scatter(points[focus, 0], points[focus, 1], s=85, color="#dc2626", zorder=3, label="focus point")
    for index in neighbors[focus]:
        axes[0].plot([points[focus, 0], points[index, 0]], [points[focus, 1], points[index, 1]], color="#2563eb", lw=2)
        axes[0].scatter(points[index, 0], points[index, 1], s=58, color="#2563eb", zorder=3)
    axes[0].set_title("Local k-nearest neighborhood", weight="bold")
    axes[0].legend(frameon=False)

    axes[1].scatter(points[:, 0], points[:, 1], s=42, color="#94a3b8")
    for source, row in enumerate(neighbors):
        for target in row[1:]:
            distance = np.linalg.norm(points[source] - points[target])
            axes[1].plot([points[source, 0], points[target, 0]], [points[source, 1], points[target, 1]], color="#2563eb", alpha=max(0.12, 0.8 - distance / 3), lw=1.2)
    axes[1].set_title("Weighted neighborhood graph", weight="bold")
    for axis in axes:
        axis.set_aspect("equal")
        axis.axis("off")
    save_figure(figure, "08_umap_neighborhood_graph.png")


def get_umap_digits_data():
    """Return a standardized digit dataset for consistent UMAP comparisons."""
    features, labels = load_digits(return_X_y=True)
    return StandardScaler().fit_transform(features), labels


def draw_embedding(axis, embedding, labels, title):
    """Draw a compact, consistently styled embedding plot."""
    scatter = axis.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=10, alpha=0.78, linewidths=0)
    axis.set_title(title, weight="bold")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return scatter


def create_umap_pca_comparison() -> None:
    """Compare linear PCA and non-linear UMAP on the same digit data."""
    features, labels = get_umap_digits_data()
    pca_embedding = PCA(n_components=2, random_state=42).fit_transform(features)
    umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(features)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    scatter = draw_embedding(axes[0], pca_embedding, labels, "PCA: linear 2D projection")
    draw_embedding(axes[1], umap_embedding, labels, "UMAP: local neighborhoods in 2D")
    figure.colorbar(scatter, ax=axes, ticks=range(10), label="Digit label", shrink=0.82)
    save_figure(figure, "08_umap_pca_comparison.png", w_pad=3.0)


def create_umap_neighbors_grid() -> None:
    """Show how n_neighbors shifts UMAP from local to broader structure."""
    features, labels = get_umap_digits_data()
    neighbor_counts = [5, 15, 50, 100]
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 7.2))
    for axis, neighbor_count in zip(axes.flat, neighbor_counts):
        embedding = umap.UMAP(n_neighbors=neighbor_count, min_dist=0.1, random_state=42).fit_transform(features)
        draw_embedding(axis, embedding, labels, f"n_neighbors = {neighbor_count}")
    figure.suptitle("UMAP parameter sensitivity: neighborhood size", weight="bold", y=0.98)
    save_figure(figure, "08_umap_neighbors_grid.png", w_pad=2.4)


def create_umap_min_dist_grid() -> None:
    """Show how min_dist controls the compactness of embedded groups."""
    features, labels = get_umap_digits_data()
    minimum_distances = [0.0, 0.1, 0.3, 0.7]
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 7.2))
    for axis, minimum_distance in zip(axes.flat, minimum_distances):
        embedding = umap.UMAP(n_neighbors=15, min_dist=minimum_distance, random_state=42).fit_transform(features)
        draw_embedding(axis, embedding, labels, f"min_dist = {minimum_distance}")
    figure.suptitle("UMAP parameter sensitivity: minimum distance", weight="bold", y=0.98)
    save_figure(figure, "08_umap_min_dist_grid.png", w_pad=2.4)


def create_umap_trustworthiness() -> None:
    """Measure neighborhood preservation across n_neighbors settings."""
    features, labels = get_umap_digits_data()
    neighbor_counts = [5, 10, 15, 30, 50, 100]
    scores = []
    for neighbor_count in neighbor_counts:
        embedding = umap.UMAP(n_neighbors=neighbor_count, min_dist=0.1, random_state=42).fit_transform(features)
        scores.append(trustworthiness(features, embedding, n_neighbors=10))

    figure, axis = plt.subplots(figsize=(8.6, 4.3))
    axis.plot(neighbor_counts, scores, color="#2563eb", marker="o", lw=2.5)
    for neighbor_count, score in zip(neighbor_counts, scores):
        axis.annotate(f"{score:.3f}", (neighbor_count, score), xytext=(0, 9), textcoords="offset points", ha="center", color="#1d4ed8", fontsize=9)
    axis.set(
        title="Neighborhood preservation across UMAP settings",
        xlabel="UMAP n_neighbors",
        ylabel="Trustworthiness (higher is better)",
        ylim=(0.75, 1.01),
    )
    axis.grid(alpha=0.2)
    save_figure(figure, "08_umap_trustworthiness.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_pca_projection()
    create_pca_3d_to_2d()
    create_pca_scree_plot()
    create_pca_reconstruction_error()
    create_pca_project_reconstruct_flow()
    create_lda_separation()
    create_random_projection()
    create_umap_neighborhoods()
    create_umap_pca_comparison()
    create_umap_neighbors_grid()
    create_umap_min_dist_grid()
    create_umap_trustworthiness()
    print("Created DS08 diagrams in fig/diagrams")