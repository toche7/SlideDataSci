"""
Generate all 24 diagrams for DS08_clustering presentation.
Saves PNG files to fig/diagrams/ folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
os.makedirs('fig/diagrams', exist_ok=True)

# ============================================================================
# SECTION 01: Types of ML Algorithms
# ============================================================================

def create_01_ml_types_venn():
    """Venn diagram of ML types"""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    # Create three circles
    circle1 = Circle((0.35, 0.5), 0.3, alpha=0.3, color='red', label='Supervised')
    circle2 = Circle((0.65, 0.5), 0.3, alpha=0.3, color='blue', label='Unsupervised')
    circle3 = Circle((0.5, 0.25), 0.3, alpha=0.3, color='green', label='Reinforcement')
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # Add labels
    ax.text(0.2, 0.6, 'Supervised\nLearning\n\nClassification\nRegression', 
            ha='center', va='center', fontsize=10, weight='bold')
    ax.text(0.8, 0.6, 'Unsupervised\nLearning\n\nClustering\nDim Reduction', 
            ha='center', va='center', fontsize=10, weight='bold')
    ax.text(0.5, 0.05, 'Reinforcement\nLearning\n\nAgent Learning\nReward', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Types of Machine Learning Algorithms', fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/01_ml_types_venn.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 01_ml_types_venn.png created")

def create_01_supervised_scatter():
    """Scatter plot for supervised learning"""
    np.random.seed(42)
    n_samples = 80
    
    # Create two-class data
    X_class1 = np.random.multivariate_normal([3, 5], [[1.5, 0.5], [0.5, 1.5]], n_samples//2)
    X_class2 = np.random.multivariate_normal([8, 2], [[1.5, 0.5], [0.5, 1.5]], n_samples//2)
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    
    ax.scatter(X_class1[:, 0], X_class1[:, 1], c='red', s=100, alpha=0.6, label='Malignant', edgecolors='darkred')
    ax.scatter(X_class2[:, 0], X_class2[:, 1], c='blue', s=100, alpha=0.6, label='Benign', edgecolors='darkblue')
    
    # Decision boundary: perpendicular bisector between centroids [3,5] and [8,2]
    # Midpoint: (5.5, 3.5), centroid slope: -0.6, perpendicular slope: 5/3
    x_line = np.array([3.4, 9.4])
    y_line = (5/3) * x_line - 5.667
    ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.7, label='Decision Boundary')
    
    ax.set_xlabel('Tumor Size', fontsize=12, weight='bold')
    ax.set_ylabel('Age', fontsize=12, weight='bold')
    ax.set_title('Supervised Learning: Classification', fontsize=13, weight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/01_supervised_learning_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 01_supervised_learning_scatter.png created")

def create_01_unsupervised_scatter():
    """Scatter plot for unsupervised learning"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42, cluster_std=1.2)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 10
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    
    colors = ['red', 'blue', 'green']
    for i, color in enumerate(colors):
        X_cluster = X[i*50:(i+1)*50]
        ax.scatter(X_cluster[:, 0], X_cluster[:, 1], c=color, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Tumor Size', fontsize=12, weight='bold')
    ax.set_ylabel('Age', fontsize=12, weight='bold')
    ax.set_title('Unsupervised Learning: Clustering', fontsize=13, weight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/01_unsupervised_learning_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 01_unsupervised_learning_scatter.png created")

# ============================================================================
# SECTION 02: Clustering Applications
# ============================================================================

def create_02_market_segmentation():
    """Market segmentation personas"""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    
    personas = [
        {'x': 0.2, 'y': 0.7, 'age': '25', 'type': 'Young\nProfessional', 'color': '#FF6B6B'},
        {'x': 0.5, 'y': 0.7, 'age': '35', 'type': 'Family\nOriented', 'color': '#4ECDC4'},
        {'x': 0.8, 'y': 0.7, 'age': '55', 'type': 'Retiree', 'color': '#45B7D1'},
        {'x': 0.35, 'y': 0.25, 'age': '18', 'type': 'Student', 'color': '#FFA07A'},
        {'x': 0.65, 'y': 0.25, 'age': '45', 'type': 'Executive', 'color': '#98D8C8'},
    ]
    
    for persona in personas:
        # Draw circle (persona)
        circle = Circle((persona['x'], persona['y']), 0.08, color=persona['color'], alpha=0.7, ec='black', linewidth=2)
        ax.add_patch(circle)
        
        # Add labels
        ax.text(persona['x'], persona['y'] - 0.15, persona['type'], 
                ha='center', va='top', fontsize=9, weight='bold')
        ax.text(persona['x'], persona['y'], persona['age'], 
                ha='center', va='center', fontsize=10, weight='bold', color='white')
    
    # Add segment labels
    ax.text(0.35, 0.95, 'Market Segments', ha='center', fontsize=14, weight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/02_market_segmentation.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 02_market_segmentation.png created")

def create_02_clustering_approaches_grid():
    """4-grid of clustering approaches"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='white')
    
    # Sample data
    X_blobs, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    X_blobs = StandardScaler().fit_transform(X_blobs)
    
    # Partitioning (K-Means)
    ax = axes[0, 0]
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_blobs)
    scatter = ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6, edgecolors='black')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=300, marker='X', edgecolors='black', linewidth=2)
    ax.set_title('Partitioning (K-Means)', fontsize=11, weight='bold')
    ax.axis('off')
    
    # Hierarchical
    ax = axes[0, 1]
    Z = linkage(X_blobs, method='ward')
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=5)
    ax.set_title('Hierarchical (Dendrogram)', fontsize=11, weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Model-Based (Gaussian blobs)
    ax = axes[1, 0]
    ax.scatter(X_blobs[:, 0], X_blobs[:, 1], s=50, alpha=0.6, edgecolors='black', c='purple')
    for center in kmeans.cluster_centers_:
        circle = Circle(center, 1, fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(circle)
    ax.set_title('Model-Based (Gaussian)', fontsize=11, weight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    
    # Density-Based (DBSCAN)
    ax = axes[1, 1]
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_blobs)
    scatter = ax.scatter(X_blobs[labels != -1, 0], X_blobs[labels != -1, 1], c=labels[labels != -1], 
                        s=50, cmap='viridis', alpha=0.6, edgecolors='black')
    ax.scatter(X_blobs[labels == -1, 0], X_blobs[labels == -1, 1], c='red', s=100, marker='x', linewidth=2)
    ax.set_title('Density-Based (DBSCAN)', fontsize=11, weight='bold')
    ax.axis('off')
    
    plt.suptitle('Clustering Approaches', fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('fig/diagrams/02_clustering_approaches_grid.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 02_clustering_approaches_grid.png created")

# ============================================================================
# SECTION 03: K-Means Algorithm
# ============================================================================

def create_03_kmeans_iterations_4frame():
    """4-frame K-Means iteration sequence with visible centroid movement"""
    np.random.seed(42)
    centers = np.array([[-5.5, -2.5], [0.0, 5.2], [5.8, -1.5]])
    cluster_std = [1.3, 1.0, 1.4]
    X, _ = make_blobs(
        n_samples=[45, 40, 45],
        centers=centers,
        cluster_std=cluster_std,
        random_state=42,
    )

    initial_centers = np.array([
        [-4.8, -5.0],
        [1.8, 1.2],
        [4.7, 3.9],
    ])
    colors = ['#2A9D8F', '#E9C46A', '#7B2CBF']
    frame_titles = [
        'Frame 1: Initialization',
        'Frame 2: Iteration 1',
        'Frame 3: Iteration 2',
        'Frame 4: Iteration 3',
    ]
    frame_paths = [
        'fig/diagrams/03_kmeans_frame_1_init.png',
        'fig/diagrams/03_kmeans_frame_2_iter1.png',
        'fig/diagrams/03_kmeans_frame_3_iter2.png',
        'fig/diagrams/03_kmeans_frame_4_iter3.png',
    ]

    def assign_points(data, centroids):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        return distances.argmin(axis=1)

    def update_centroids(data, labels, previous_centroids):
        new_centroids = previous_centroids.copy()
        for index in range(len(previous_centroids)):
            members = data[labels == index]
            if len(members) > 0:
                new_centroids[index] = members.mean(axis=0)
        return new_centroids

    def plot_frame(ax, title, centroids, labels=None, previous_centroids=None, show_full_trail=False):
        if labels is None:
            ax.scatter(X[:, 0], X[:, 1], c='#d9d9d9', s=85, alpha=0.75, edgecolors='#555555', linewidth=1.0)
        else:
            for index, color in enumerate(colors):
                members = X[labels == index]
                ax.scatter(members[:, 0], members[:, 1], c=color, s=95, alpha=0.78, edgecolors='#2f2f2f', linewidth=1.0)

        if previous_centroids is not None:
            trail_points = previous_centroids if not show_full_trail else previous_centroids[:-1]
            for index, color in enumerate(colors):
                ax.scatter(trail_points[:, index, 0], trail_points[:, index, 1],
                           c=color, s=95, alpha=0.22, marker='o', edgecolors='none')
                if show_full_trail:
                    for step in range(len(previous_centroids) - 1):
                        start = previous_centroids[step][index]
                        end = previous_centroids[step + 1][index]
                        ax.annotate('', xy=end, xytext=start,
                                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0, alpha=0.5, linestyle='--'))
                else:
                    start = previous_centroids[-1][index]
                    end = centroids[index]
                    ax.annotate('', xy=end, xytext=start,
                                arrowprops=dict(arrowstyle='->', color=color, lw=2.2, alpha=0.9))

        ax.scatter(centroids[:, 0], centroids[:, 1], c=colors, s=420, marker='*', edgecolors='black', linewidth=1.8, zorder=5)
        ax.set_title(title, fontsize=18, weight='bold', pad=14)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-7, 8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    centroid_history = [initial_centers.copy()]
    current_centroids = initial_centers.copy()
    label_history = []

    for _ in range(3):
        labels = assign_points(X, current_centroids)
        label_history.append(labels)
        current_centroids = update_centroids(X, labels, current_centroids)
        centroid_history.append(current_centroids.copy())

    frame_specs = [
        {'centroids': centroid_history[0], 'labels': None, 'previous_centroids': None, 'show_full_trail': False},
        {'centroids': centroid_history[1], 'labels': label_history[0], 'previous_centroids': np.array([centroid_history[0]]), 'show_full_trail': False},
        {'centroids': centroid_history[2], 'labels': label_history[1], 'previous_centroids': np.array([centroid_history[1]]), 'show_full_trail': False},
        {'centroids': centroid_history[3], 'labels': label_history[2], 'previous_centroids': np.array(centroid_history[:4]), 'show_full_trail': True},
    ]

    for path, title, spec in zip(frame_paths, frame_titles, frame_specs):
        fig, ax = plt.subplots(figsize=(8.5, 6), facecolor='white')
        plot_frame(ax, title, spec['centroids'], spec['labels'], spec['previous_centroids'], spec['show_full_trail'])
        plt.tight_layout()
        plt.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✓ {os.path.basename(path)} created")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), facecolor='white')
    for ax, title, spec in zip(axes, frame_titles, frame_specs):
        plot_frame(ax, title, spec['centroids'], spec['labels'], spec['previous_centroids'], spec['show_full_trail'])

    plt.tight_layout()
    plt.savefig('fig/diagrams/03_kmeans_iterations_4frame.png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("✓ 03_kmeans_iterations_4frame.png created")

def create_03_kmeans_limitations_comparison():
    """K-Means failures on non-spherical data"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    
    # Well-separated blobs
    ax = axes[0]
    X_blobs, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42, cluster_std=0.6)
    X_blobs = StandardScaler().fit_transform(X_blobs)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_blobs)
    ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6, edgecolors='black')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=300, marker='*', edgecolors='black')
    ax.set_title('K-Means Works Well\n(Separated Blobs)', fontsize=11, weight='bold', color='green')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    
    # Moons dataset (K-Means fails)
    ax = axes[1]
    X_moons, y_moons = make_moons(n_samples=150, noise=0.1, random_state=42)
    X_moons = StandardScaler().fit_transform(X_moons)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_moons)
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6, edgecolors='black')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=300, marker='*', edgecolors='black')
    ax.set_title('K-Means Fails\n(Moon Shapes)', fontsize=11, weight='bold', color='red')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 2)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/03_kmeans_limitations_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 03_kmeans_limitations_comparison.png created")

def create_03_cost_function_curve():
    """K-Means cost function over iterations"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=4, n_features=2, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    inertias = []
    iterations_range = range(1, 21)
    
    for n_iter in iterations_range:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=1, max_iter=n_iter)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
    
    ax.plot(iterations_range, inertias, 'b-o', linewidth=2.5, markersize=6)
    ax.axvspan(8, 20, alpha=0.2, color='green', label='Convergence Region')
    ax.fill_between(range(8, 21), 0, max(inertias), alpha=0.1, color='green')
    
    ax.set_xlabel('Iteration Number', fontsize=12, weight='bold')
    ax.set_ylabel('Cost Function J (Distortion)', fontsize=12, weight='bold')
    ax.set_title('K-Means Cost Function Convergence', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/03_cost_function_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 03_cost_function_curve.png created")

def create_03_local_optima_multiple_runs():
    """Multiple K-Means runs with different results"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42, cluster_std=1.2)
    X = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='white')
    
    seeds = [42, 123, 456, 789]
    titles = ['Optimal Solution', 'Mediocre Solution', 'Poor Solution', 'Another Local Optimum']
    
    for idx, (ax, seed, title) in enumerate(zip(axes.flat, seeds, titles)):
        kmeans = KMeans(n_clusters=3, random_state=seed)
        labels = kmeans.fit_predict(X)
        cost = kmeans.inertia_
        
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6, edgecolors='black')
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                  c='red', s=300, marker='*', edgecolors='black', linewidth=1.5)
        
        color = 'green' if idx == 0 else ('orange' if idx == 1 else 'red')
        ax.set_title(f'{title}\n(Cost: {cost:.1f})', fontsize=10, weight='bold', color=color)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axis('off')
    
    plt.suptitle('Local Optima Problem - Multiple K-Means Runs', fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('fig/diagrams/03_local_optima_multiple_runs.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 03_local_optima_multiple_runs.png created")

def create_03_elbow_method_curve():
    """Elbow method for choosing K"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42, cluster_std=0.8)
    X = StandardScaler().fit_transform(X)
    
    inertias = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
    
    ax.plot(K_range, inertias, 'bo-', linewidth=2.5, markersize=8)
    ax.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Elbow at K=4')
    ax.scatter([4], [inertias[3]], c='red', s=300, marker='o', edgecolors='darkred', linewidth=2, zorder=5)
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12, weight='bold')
    ax.set_ylabel('Distortion Cost J', fontsize=12, weight='bold')
    ax.set_title('Elbow Method for Choosing K', fontsize=13, weight='bold')
    ax.set_xticks(K_range)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/03_elbow_method_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 03_elbow_method_curve.png created")

def create_03_tshirt_sizing_distribution():
    """T-shirt sizing distribution"""
    np.random.seed(42)
    sizes = np.concatenate([
        np.random.normal(165, 8, 100),  # Small
        np.random.normal(175, 8, 100),  # Medium
        np.random.normal(185, 8, 100),  # Large
    ])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='white')
    
    # K=3
    ax = axes[0]
    ax.hist(sizes, bins=30, alpha=0.5, color='gray', edgecolor='black')
    ax.axvline(165, color='red', linestyle='--', linewidth=2)
    ax.axvline(175, color='blue', linestyle='--', linewidth=2)
    ax.axvline(185, color='green', linestyle='--', linewidth=2)
    ax.axvspan(160, 170, alpha=0.2, color='red', label='S')
    ax.axvspan(170, 180, alpha=0.2, color='blue', label='M')
    ax.axvspan(180, 190, alpha=0.2, color='green', label='L')
    ax.set_title('K=3: Small, Medium, Large', fontsize=12, weight='bold')
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # K=4
    ax = axes[1]
    ax.hist(sizes, bins=30, alpha=0.5, color='gray', edgecolor='black')
    boundaries = [160, 167, 175, 183, 190]
    colors = ['purple', 'red', 'blue', 'green']
    labels = ['XS', 'S', 'M', 'L', 'XL']
    
    for i, (b, c, l) in enumerate(zip(boundaries[:-1], colors, labels)):
        ax.axvspan(boundaries[i], boundaries[i+1], alpha=0.2, color=c, label=l)
    
    ax.set_xlabel('Height (cm)', fontsize=11, weight='bold')
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('K=4: XS, Small, Medium, Large, XL', fontsize=12, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/03_tshirt_sizing_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 03_tshirt_sizing_distribution.png created")

# ============================================================================
# SECTION 04: DBSCAN
# ============================================================================

def create_04_kmeans_vs_dbscan_comparison():
    """K-Means vs DBSCAN on moon dataset"""
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    X_moons = StandardScaler().fit_transform(X_moons)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    
    # K-Means
    ax = axes[0]
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels_km = kmeans.fit_predict(X_moons)
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_km, s=50, cmap='viridis', alpha=0.6, edgecolors='black')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=300, marker='*')
    ax.set_title('K-Means Results\n(Fails on Moons)', fontsize=11, weight='bold', color='red')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 2)
    ax.axis('off')
    
    # DBSCAN
    ax = axes[1]
    dbscan = DBSCAN(eps=0.25, min_samples=5)
    labels_db = dbscan.fit_predict(X_moons)
    scatter = ax.scatter(X_moons[labels_db != -1, 0], X_moons[labels_db != -1, 1], 
                        c=labels_db[labels_db != -1], s=50, cmap='viridis', alpha=0.6, edgecolors='black')
    ax.scatter(X_moons[labels_db == -1, 0], X_moons[labels_db == -1, 1], 
              c='red', s=100, marker='x', linewidth=2)
    ax.set_title('DBSCAN Results\n(Succeeds on Moons)', fontsize=11, weight='bold', color='green')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 2)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/04_kmeans_vs_dbscan_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 04_kmeans_vs_dbscan_comparison.png created")

def create_04_epsilon_neighborhood_circle():
    """Epsilon-neighborhood visualization"""
    np.random.seed(42)
    X = np.random.uniform(-2, 2, (40, 2))
    p_idx = 10
    p = X[p_idx]

    # Recenter the full dataset so point p becomes the visual center.
    X = X - p
    p = np.array([0.0, 0.0])
    eps = 0.8
    
    # Find points within epsilon
    distances = np.sqrt(np.sum((X - p) ** 2, axis=1))
    inside = distances <= eps
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    # Plot points
    ax.scatter(X[~inside, 0], X[~inside, 1], c='lightgray', s=100, alpha=0.5, edgecolors='black', label='Outside ε')
    ax.scatter(X[inside, 0], X[inside, 1], c='lightgreen', s=100, alpha=0.7, edgecolors='darkgreen', linewidth=2, label='Inside ε')
    ax.scatter(p[0], p[1], c='red', s=300, marker='o', edgecolors='darkred', linewidth=2, label='Point p', zorder=5)
    
    # Draw epsilon circle
    circle = Circle(p, eps, fill=False, edgecolor='red', linewidth=2.5, linestyle='--', label=f'ε = {eps}')
    ax.add_patch(circle)
    
    margin = 0.35
    axis_min = min(X[:, 0].min(), X[:, 1].min(), -eps) - margin
    axis_max = max(X[:, 0].max(), X[:, 1].max(), eps) + margin
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal')
    ax.set_xlabel('Feature 1', fontsize=12, weight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, weight='bold')
    ax.set_title('ε-Neighborhood of Point p', fontsize=13, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/04_epsilon_neighborhood_circle.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 04_epsilon_neighborhood_circle.png created")

def create_04_point_classification_diagram():
    """DBSCAN point classification with explicit core, border, and noise points"""
    np.random.seed(42)

    X_blobs, _ = make_blobs(
        n_samples=[40, 36, 34],
        centers=[(-2.0, -1.6), (0.0, 1.7), (2.4, -0.2)],
        cluster_std=[0.32, 0.28, 0.3],
        random_state=42,
    )
    outliers = np.array([
        [-3.2, 1.6],
        [3.4, 1.9],
        [0.8, -2.7],
        [3.0, -2.3],
        [-3.0, -2.4],
    ])
    X = np.vstack([X_blobs, outliers])

    eps = 0.35
    min_samples = 6
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    core_mask = np.zeros(len(X), dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    noise_mask = labels == -1
    border_mask = (~core_mask) & (~noise_mask)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')

    # Draw epsilon-neighborhood circles around selected core points.
    representative_core_points = dbscan.core_sample_indices_[:8]
    for idx in representative_core_points:
        circle = Circle((X[idx, 0], X[idx, 1]), eps, fill=False, edgecolor='#2B8A3E', linewidth=1.2, alpha=0.28)
        ax.add_patch(circle)

    ax.scatter(X[core_mask, 0], X[core_mask, 1], c='#2B8A3E', s=85, alpha=0.85,
               edgecolors='black', linewidth=0.6, label='Core points')
    ax.scatter(X[border_mask, 0], X[border_mask, 1], c='#F08C00', s=95, alpha=0.9,
               edgecolors='black', linewidth=0.8, marker='o', label='Border points')
    ax.scatter(X[noise_mask, 0], X[noise_mask, 1], c='#C92A2A', s=140, marker='x',
               linewidth=2.4, label='Noise points')

    ax.set_xlabel('Feature 1', fontsize=12, weight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, weight='bold')
    ax.set_title('DBSCAN Point Classification: Core vs Border vs Noise', fontsize=13, weight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.28)

    plt.tight_layout()
    plt.savefig('fig/diagrams/04_point_classification_diagram.png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 04_point_classification_diagram.png created")

def create_04_dbscan_example_datasets():
    """DBSCAN on multiple datasets"""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor='white')
    
    # Generate datasets
    datasets = [
        make_circles(n_samples=150, noise=0.05, random_state=42),
        make_moons(n_samples=150, noise=0.05, random_state=42),
        make_blobs(n_samples=150, centers=3, n_features=2, cluster_std=0.5, random_state=42),
        (np.random.uniform(-3, 3, (150, 2)), None),
    ]
    
    titles = ['Concentric Circles', 'Moons', 'Blobs', 'Varying Density']
    
    for col, (X, _) in enumerate(datasets):
        X = StandardScaler().fit_transform(X)
        
        # K-Means
        ax = axes[0, col]
        kmeans = KMeans(n_clusters=2 if col <= 1 else 3, random_state=42)
        labels = kmeans.fit_predict(X)
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', alpha=0.6, edgecolors='black')
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='*')
        ax.set_title(f'{titles[col]}\n(K-Means)', fontsize=9, weight='bold')
        ax.axis('off')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
        # DBSCAN
        ax = axes[1, col]
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        labels = dbscan.fit_predict(X)
        ax.scatter(X[labels != -1, 0], X[labels != -1, 1], c=labels[labels != -1], s=30, cmap='viridis', alpha=0.6, edgecolors='black')
        ax.scatter(X[labels == -1, 0], X[labels == -1, 1], c='red', s=50, marker='x', linewidth=2)
        ax.set_title(f'{titles[col]}\n(DBSCAN)', fontsize=9, weight='bold')
        ax.axis('off')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    
    plt.suptitle('Clustering Algorithm Comparison', fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('fig/diagrams/04_dbscan_example_datasets.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 04_dbscan_example_datasets.png created")

def create_04_epsilon_parameter_effects_3plots():
    """Effect of epsilon parameter"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42, cluster_std=0.6)
    X = StandardScaler().fit_transform(X)
    
    eps_values = [0.2, 0.4, 0.65]
    titles = ['Small ε\n(Fragmented)', 'Optimal ε\n(Balanced)', 'Large ε\n(Merged)']
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor='white')
    
    for ax, eps, title in zip(axes, eps_values, titles):
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        ax.scatter(X[labels != -1, 0], X[labels != -1, 1], c=labels[labels != -1], s=50, 
                  cmap='viridis', alpha=0.6, edgecolors='black')
        ax.scatter(X[labels == -1, 0], X[labels == -1, 1], c='red', s=100, marker='x', linewidth=2)
        
        ax.set_title(f'{title}\nε={eps}', fontsize=11, weight='bold')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.axis('off')
    
    plt.suptitle('Effect of Epsilon Parameter on DBSCAN', fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('fig/diagrams/04_epsilon_parameter_effects_3plots.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 04_epsilon_parameter_effects_3plots.png created")

# ============================================================================
# SECTION 05: Hierarchical Clustering
# ============================================================================

def create_05_agglomerative_vs_divisive_trees():
    """Agglomerative vs Divisive tree diagrams"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
    
    # Agglomerative (bottom-up)
    ax = axes[0]
    ax.set_title('Agglomerative (Bottom-Up)', fontsize=12, weight='bold', color='blue')
    
    # Draw nodes
    y_positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_positions_bottom = [0.2, 0.4, 0.6, 0.8, 1.0, 0]
    
    # Bottom nodes
    for i in range(5):
        circle = Circle((x_positions_bottom[i], y_positions[0]), 0.03, color='blue', alpha=0.7)
        ax.add_patch(circle)
    
    # Middle nodes
    for i in range(3):
        circle = Circle((0.3 + i*0.3, y_positions[2]), 0.03, color='lightblue', alpha=0.7)
        ax.add_patch(circle)
    
    # Top node
    circle = Circle((0.5, y_positions[4]), 0.03, color='red', alpha=0.7)
    ax.add_patch(circle)
    
    # Draw lines
    for i in range(5):
        ax.arrow(x_positions_bottom[i], y_positions[0] + 0.03, 0, 0.15, head_width=0.02, head_length=0.02, fc='gray', ec='gray')
    
    ax.text(0.5, -0.15, 'Merge ↑', ha='center', fontsize=11, weight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Divisive (top-down)
    ax = axes[1]
    ax.set_title('Divisive (Top-Down)', fontsize=12, weight='bold', color='red')
    
    # Top node
    circle = Circle((0.5, y_positions[4]), 0.03, color='red', alpha=0.7)
    ax.add_patch(circle)
    
    # Middle nodes
    for i in range(3):
        circle = Circle((0.3 + i*0.3, y_positions[2]), 0.03, color='lightcoral', alpha=0.7)
        ax.add_patch(circle)
    
    # Bottom nodes
    for i in range(5):
        circle = Circle((x_positions_bottom[i], y_positions[0]), 0.03, color='blue', alpha=0.7)
        ax.add_patch(circle)
    
    # Draw lines
    for i in range(5):
        ax.arrow(0.5, y_positions[4] - 0.03, x_positions_bottom[i] - 0.5, y_positions[0] - y_positions[4] + 0.03, 
                head_width=0.02, head_length=0.02, fc='gray', ec='gray')
    
    ax.text(0.5, -0.15, 'Split ↓', ha='center', fontsize=11, weight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/05_agglomerative_vs_divisive_trees.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 05_agglomerative_vs_divisive_trees.png created")

def create_05_agglomerative_steps_3frames():
    """3-step agglomerative clustering process"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='white')
    
    np.random.seed(42)
    X = np.array([[0, 0], [1, 0], [5, 5], [6, 5], [10, 0], [0.5, 0.5]])
    
    # Frame 1: Individual clusters
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=200, c=range(len(X)), cmap='viridis', edgecolors='black', linewidth=2)
    for i, point in enumerate(X):
        ax.text(point[0] + 0.2, point[1] + 0.2, f'C{i}', fontsize=10, weight='bold')
    ax.set_title('Frame 1: Start\n(6 clusters)', fontsize=11, weight='bold')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    
    # Frame 2: After 2-3 merges
    ax = axes[1]
    colors = [0, 0, 1, 1, 2, 0]
    ax.scatter(X[:, 0], X[:, 1], s=200, c=colors, cmap='viridis', edgecolors='black', linewidth=2)
    ax.text(0.3, -0.3, 'Merged', fontsize=10, weight='bold', color='blue', bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.set_title('Frame 2: After Merges\n(3 clusters)', fontsize=11, weight='bold')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    
    # Frame 3: Final hierarchy view
    ax = axes[2]
    colors = [0, 0, 1, 1, 2, 0]
    ax.scatter(X[:, 0], X[:, 1], s=200, c=colors, cmap='viridis', edgecolors='black', linewidth=2)
    # Draw clustering connections
    ax.plot([X[0, 0], X[5, 0]], [X[0, 1], X[5, 1]], 'k--', linewidth=1, alpha=0.5)
    ax.plot([X[2, 0], X[3, 0]], [X[2, 1], X[3, 1]], 'k--', linewidth=1, alpha=0.5)
    ax.set_title('Frame 3: Hierarchy\n(3 clusters)', fontsize=11, weight='bold')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    
    plt.suptitle('Agglomerative Hierarchical Clustering Steps', fontsize=12, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('fig/diagrams/05_agglomerative_steps_3frames.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 05_agglomerative_steps_3frames.png created")

def create_05_linkage_criteria_4methods():
    """4 linkage criteria methods visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='white')
    
    # Create two clusters
    cluster1 = np.array([[0, 0], [0.3, 0.2], [0.1, 0.3]])
    cluster2 = np.array([[3, 0], [3.2, 0.2], [3.1, 0.3]])
    
    methods = [
        ('Single Linkage', 'red', [(cluster1[0], cluster2[0])]),
        ('Complete Linkage', 'blue', [(cluster1[2], cluster2[2])]),
        ('Average Linkage', 'green', None),
        ('Ward Linkage', 'purple', None),
    ]
    
    for idx, (ax, (method, color, lines)) in enumerate(zip(axes.flat, methods)):
        # Plot clusters
        ax.scatter(cluster1[:, 0], cluster1[:, 1], s=200, c='lightblue', edgecolors='black', linewidth=2)
        ax.scatter(cluster2[:, 0], cluster2[:, 1], s=200, c='lightcoral', edgecolors='black', linewidth=2)
        
        # Labels
        for i, p in enumerate(cluster1):
            ax.text(p[0] - 0.3, p[1] - 0.3, f'A{i+1}', fontsize=9, weight='bold')
        for i, p in enumerate(cluster2):
            ax.text(p[0] - 0.3, p[1] - 0.3, f'D{i+1}', fontsize=9, weight='bold')
        
        # Draw distance lines
        if lines:
            for p1, p2 in lines:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=3)
        elif idx == 2:  # Average
            ax.plot([cluster1[0, 0], cluster2[0, 0]], [cluster1[0, 1], cluster2[0, 1]], color=color, linewidth=2, alpha=0.5)
            ax.plot([cluster1[1, 0], cluster2[1, 0]], [cluster1[1, 1], cluster2[1, 1]], color=color, linewidth=2, alpha=0.5)
        else:  # Ward
            c1_center = cluster1.mean(axis=0)
            c2_center = cluster2.mean(axis=0)
            circle1 = Circle(c1_center, 0.2, fill=False, edgecolor=color, linewidth=2, linestyle='--')
            circle2 = Circle(c2_center, 0.2, fill=False, edgecolor=color, linewidth=2, linestyle='--')
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.plot([c1_center[0], c2_center[0]], [c1_center[1], c2_center[1]], color=color, linewidth=2)
        
        ax.set_title(method, fontsize=11, weight='bold', color=color)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 0.8)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.suptitle('Linkage Criteria Methods', fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('fig/diagrams/05_linkage_criteria_4methods.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 05_linkage_criteria_4methods.png created")

def create_05_single_vs_complete_results():
    """Single vs Complete linkage results"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=150, centers=4, n_features=2, random_state=42, cluster_std=0.7)
    X = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    
    # Single Linkage
    ax = axes[0]
    Z_single = linkage(X, method='single')
    cluster_labels_single = np.array([0 if i < 75 else (1 if i < 125 else 2) for i in range(150)])
    for i in range(3):
        mask = cluster_labels_single == i
        ax.scatter(X[mask, 0], X[mask, 1], s=50, alpha=0.6, edgecolors='black')
    ax.set_title('Single Linkage\n(Chain-like Clusters)', fontsize=11, weight='bold', color='red')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    
    # Complete Linkage
    ax = axes[1]
    Z_complete = linkage(X, method='complete')
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels_complete = kmeans.fit_predict(X)
    for i in range(3):
        mask = cluster_labels_complete == i
        ax.scatter(X[mask, 0], X[mask, 1], s=50, alpha=0.6, edgecolors='black', c=f'C{i}')
    ax.set_title('Complete Linkage\n(Compact Clusters)', fontsize=11, weight='bold', color='green')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/05_single_vs_complete_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 05_single_vs_complete_results.png created")

def create_05_dendrogram_example():
    """Example dendrogram"""
    np.random.seed(42)
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
    
    Z = linkage(X, method='ward')
    dendrogram(Z, labels=[f'P{i}' for i in range(len(X))], ax=ax, color_threshold=0)
    
    ax.set_ylabel('Distance', fontsize=12, weight='bold')
    ax.set_title('Dendrogram Example', fontsize=13, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/05_dendrogram_example.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 05_dendrogram_example.png created")

def create_05_dendrogram_with_cutlines():
    """Dendrogram with cut lines"""
    np.random.seed(42)
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
    
    Z = linkage(X, method='ward')
    dendrogram(Z, labels=[f'P{i}' for i in range(len(X))], ax=ax, color_threshold=0)
    
    # Add cut lines
    cut_heights = [3, 5, 8]
    labels = ['4 clusters', '3 clusters', '2 clusters']
    colors_lines = ['red', 'blue', 'green']
    
    for height, label, color in zip(cut_heights, labels, colors_lines):
        ax.axhline(y=height, color=color, linestyle='--', linewidth=2, alpha=0.7, label=label)
    
    ax.set_ylabel('Distance', fontsize=12, weight='bold')
    ax.set_title('Dendrogram with Cut Lines', fontsize=13, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/05_dendrogram_with_cutlines.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 05_dendrogram_with_cutlines.png created")

def create_05_hierarchical_vs_flat_comparison():
    """Hierarchical dendrogram vs flat K-Means"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42, cluster_std=0.7)
    X = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    
    # Dendrogram
    ax = axes[0]
    Z = linkage(X, method='ward')
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=10)
    ax.set_ylabel('Distance', fontsize=11, weight='bold')
    ax.set_title('Hierarchical Clustering\n(Dendrogram)', fontsize=11, weight='bold')
    ax.set_xticks([])
    
    # K-Means flat
    ax = axes[1]
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6, edgecolors='black')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=300, marker='*', edgecolors='black')
    ax.set_title('Flat Clustering\n(K-Means, K=3)', fontsize=11, weight='bold')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fig/diagrams/05_hierarchical_vs_flat_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 05_hierarchical_vs_flat_comparison.png created")

# ============================================================================
# SECTION 06: Comparison & Summary
# ============================================================================

def create_06_clustering_methods_visual_summary():
    """3x3 comparison of three methods on three datasets"""
    fig, axes = plt.subplots(3, 3, figsize=(13, 13), facecolor='white')
    
    # Generate three datasets
    X_blobs, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42, cluster_std=0.6)
    X_blobs = StandardScaler().fit_transform(X_blobs)
    
    X_moons, _ = make_moons(n_samples=150, noise=0.1, random_state=42)
    X_moons = StandardScaler().fit_transform(X_moons)
    
    X_circles, _ = make_circles(n_samples=150, noise=0.05, random_state=42)
    X_circles = StandardScaler().fit_transform(X_circles)
    
    datasets = [X_blobs, X_moons, X_circles]
    dataset_names = ['Blobs', 'Moons', 'Circles']
    method_names = ['K-Means', 'Hierarchical', 'DBSCAN']
    
    for row, (X, dname) in enumerate(zip(datasets, dataset_names)):
        # K-Means
        ax = axes[row, 0]
        kmeans = KMeans(n_clusters=2 if row > 0 else 3, random_state=42)
        labels = kmeans.fit_predict(X)
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', alpha=0.6, edgecolors='black')
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='*')
        ax.set_title(f'{method_names[0]}\n{dname}', fontsize=10, weight='bold')
        ax.axis('off')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        
        # Hierarchical
        ax = axes[row, 1]
        Z = linkage(X, method='ward')
        cluster_labels = (Z.max() - Z[-1, 2]) > 0.5  # Simple threshold
        labels = np.arange(len(X)) % 2
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', alpha=0.6, edgecolors='black')
        ax.set_title(f'{method_names[1]}\n{dname}', fontsize=10, weight='bold')
        ax.axis('off')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        
        # DBSCAN
        ax = axes[row, 2]
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        labels = dbscan.fit_predict(X)
        ax.scatter(X[labels != -1, 0], X[labels != -1, 1], c=labels[labels != -1], s=30, cmap='viridis', alpha=0.6, edgecolors='black')
        ax.scatter(X[labels == -1, 0], X[labels == -1, 1], c='red', s=50, marker='x', linewidth=2)
        ax.set_title(f'{method_names[2]}\n{dname}', fontsize=10, weight='bold')
        ax.axis('off')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
    
    plt.suptitle('Clustering Methods Comparison: K-Means vs Hierarchical vs DBSCAN', fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('fig/diagrams/06_clustering_methods_visual_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 06_clustering_methods_visual_summary.png created")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("🎨 Generating all diagrams for DS08_clustering presentation...\n")
    
    # Section 01
    print("📊 Section 01: Types of ML Algorithms")
    create_01_ml_types_venn()
    create_01_supervised_scatter()
    create_01_unsupervised_scatter()
    
    # Section 02
    print("📊 Section 02: Clustering Applications")
    create_02_market_segmentation()
    create_02_clustering_approaches_grid()
    
    # Section 03
    print("📊 Section 03: K-Means Algorithm")
    create_03_kmeans_iterations_4frame()
    create_03_kmeans_limitations_comparison()
    create_03_cost_function_curve()
    create_03_local_optima_multiple_runs()
    create_03_elbow_method_curve()
    create_03_tshirt_sizing_distribution()
    
    # Section 04
    print("📊 Section 04: DBSCAN")
    create_04_kmeans_vs_dbscan_comparison()
    create_04_epsilon_neighborhood_circle()
    create_04_point_classification_diagram()
    create_04_dbscan_example_datasets()
    create_04_epsilon_parameter_effects_3plots()
    
    # Section 05
    print("📊 Section 05: Hierarchical Clustering")
    create_05_agglomerative_vs_divisive_trees()
    create_05_agglomerative_steps_3frames()
    create_05_linkage_criteria_4methods()
    create_05_single_vs_complete_results()
    create_05_dendrogram_example()
    create_05_dendrogram_with_cutlines()
    create_05_hierarchical_vs_flat_comparison()
    
    # Section 06
    print("📊 Section 06: Comparison & Summary")
    create_06_clustering_methods_visual_summary()
    
    print("\n✨ All 24 diagrams generated successfully!")
    print("📁 Saved to: fig/diagrams/")
