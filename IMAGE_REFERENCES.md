# Image References for DS08_clustering.md

Complete list of images to create. All should be saved in `fig/diagrams/` folder.

---

## Section 01: Types of ML Algorithms

### 1. **01_ml_types_venn.png**
- **Type:** Venn diagram
- **Description:** Three overlapping circles labeled "Supervised Learning", "Unsupervised Learning", and "Reinforcement Learning". Include icons for each: classification/regression for supervised, clustering/dimensionality reduction for unsupervised, agent/reward for reinforcement learning.
- **Location:** Slide "Types of ML Algorithms" (right side)
- **Suggested size:** 600x600px

### 2. **01_supervised_learning_scatter.png**
- **Type:** 2D scatter plot
- **Description:** X-axis = Tumor Size, Y-axis = Age. Red dots = malignant tumors, Blue dots = benign tumors. Include a clear decision boundary line/curve separating the two classes.
- **Location:** Slide "Supervised Learning: Classification" (right side)
- **Suggested size:** 600x400px

### 3. **01_unsupervised_learning_scatter.png**
- **Type:** 2D scatter plot
- **Description:** Same axes (Tumor Size vs Age) but WITHOUT labels and WITHOUT decision boundary. Show natural groupings with different colored points representing 2-3 discovered clusters. No separation lines, just colored points.
- **Location:** Slide "Unsupervised Learning" (right side)
- **Suggested size:** 600x400px

---

## Section 02: Clustering Applications

### 4. **02_market_segmentation.png**
- **Type:** Illustration
- **Description:** 3-4 customer personas/avatars showing different demographics (young/old), shopping behaviors (casual/formal), preferences (tech/fashion). Arrange in groups or colored regions to show segmentation.
- **Location:** Slide "Example 1: Market Segmentation & Social Networks" (left side)
- **Suggested size:** 500x400px

### 5. **02_clustering_approaches_grid.png**
- **Type:** 2x2 grid of examples
- **Description:** Four subplots showing different clustering outputs:
  1. Top-left: Partitioning (K=3 spherical clusters)
  2. Top-right: Hierarchical (dendrogram tree structure)
  3. Bottom-left: Model-Based (Gaussian mixture/blob shapes)
  4. Bottom-right: Density-Based (irregular shapes with isolated noise points marked as ×)
- **Location:** Slide "Major Clustering Approaches" (right side)
- **Suggested size:** 700x600px

---

## Section 03: K-Means Algorithm

### 6. **03_kmeans_iterations_4frame.png**
- **Type:** 4-frame sequence
- **Description:** 
  - Frame 1: Scattered data points with 3 random red × markers (initial centroids)
  - Frame 2: Points colored by cluster (red/blue/green) based on nearest centroid
  - Frame 3: Centroids moved to cluster means (within-cluster centers)
  - Frame 4: Final convergence with stabilized clusters
- **Location:** Slide "K-Means Algorithm Visualization" (right side)
- **Suggested size:** 800x400px

### 7. **03_kmeans_limitations_comparison.png**
- **Type:** Side-by-side comparison (2 plots)
- **Description:**
  - Left: Well-separated spherical clusters with K-Means working perfectly
  - Right: Moon/crescent-shaped dataset showing K-Means failure with overlapping cluster boundaries
- **Location:** Slide "K-Means for Non-Separated Clusters" (right side)
- **Suggested size:** 800x400px

### 8. **03_cost_function_curve.png**
- **Type:** Line graph
- **Description:** X-axis = Iteration (0-20), Y-axis = Cost J. Show decreasing curve starting high and plateauing around iteration 8-10. Annotate plateau region with "convergence".
- **Location:** Slide "Optimization Objective" (right side)
- **Suggested size:** 600x400px

### 9. **03_local_optima_multiple_runs.png**
- **Type:** 3-4 subplots
- **Description:** Same dataset clustered 4 different ways:
  1. Optimal solution (good clustering)
  2. Mediocre solution (acceptable but suboptimal)
  3. Poor solution (bad initialization)
  4. (Optional) Cost comparison table showing different J values
  Use different colors for cluster assignments in each subplot.
- **Location:** Slide "Local Optima Problem" (right side)
- **Suggested size:** 700x600px

### 10. **03_elbow_method_curve.png**
- **Type:** Line graph
- **Description:** X-axis = Number of clusters K (1-10), Y-axis = Distortion cost J. Show rapid decrease initially (steep slope), then gradual decrease with clear elbow point marked at K=3 or K=4. Add annotation arrow pointing to "Elbow".
- **Location:** Slide "Elbow Method" (right side)
- **Suggested size:** 600x400px

### 11. **03_tshirt_sizing_distribution.png**
- **Type:** Histogram/distribution plot (2 subplots)
- **Description:**
  - Top: Histogram of customer sizes with K=3 showing 3 overlapping regions for S/M/L
  - Bottom: Same histogram with K=4 showing 5 regions for XS/S/M/L/XL
  Color-code each size region differently.
- **Location:** Slide "Practical Example: T-Shirt Sizing" (right side)
- **Suggested size:** 600x500px

---

## Section 04: Density-Based Clustering (DBSCAN)

### 12. **04_kmeans_vs_dbscan_comparison.png**
- **Type:** Split-screen comparison (2 plots)
- **Description:** Same crescent moon dataset:
  - Left: K-Means result showing bad spherical clusters cutting through the moons
  - Right: DBSCAN result correctly identifying the two crescent shapes
- **Location:** Slide "Why Density-Based Clustering?" (right side)
- **Suggested size:** 800x400px

### 13. **04_epsilon_neighborhood_circle.png**
- **Type:** Scatter plot with geometric overlay
- **Description:** Multiple points scattered on plot. Center point labeled "p" with circle of radius ε around it. Inside circle: 5-6 points highlighted in green (within ε-neighborhood). Outside circle: points grayed out (not in neighborhood).
- **Location:** Slide "Definitions: Epsilon-Neighborhood" (right side)
- **Suggested size:** 600x400px

### 14. **04_point_classification_diagram.png**
- **Type:** Scatter plot with classification
- **Description:** Show three types of DBSCAN points:
  - Core points: Solid colored circles with small circles around them (showing ε-neighborhoods)
  - Border points: On edges of core point neighborhoods, different color
  - Noise points: Isolated points marked with × symbols
  Legend explaining each type.
- **Location:** Slide "Point Classification in DBSCAN" (right side)
- **Suggested size:** 600x500px

### 15. **04_dbscan_example_datasets.png**
- **Type:** Grid of 4 datasets with side-by-side results (2x2 grid)
- **Description:** Four challenging 2D datasets showing K-Means vs DBSCAN results:
  1. Concentric circles (K-Means fails = concentric circles split badly; DBSCAN succeeds = identifies two circles correctly)
  2. Two moons (K-Means = overlapping clusters; DBSCAN = correct moons)
  3. Blobs with noise (K-Means = includes noise in clusters; DBSCAN = isolates noise points with ×)
  4. Varying density clusters (K-Means = equal-size spheres fail; DBSCAN = captures different densities)
- **Location:** Slide "DBSCAN Examples" (right side)
- **Suggested size:** 900x700px

### 16. **04_epsilon_parameter_effects_3plots.png**
- **Type:** 3-panel comparison
- **Description:** Same dataset with different ε values:
  - Left: Large ε → Many points core/border, few noise points, clusters merged together
  - Middle: Small ε → Few core/border points, many noise points (×), clusters fragmented
  - Right: Optimal ε → Well-balanced clustering, appropriate number of noise points
  Label each plot with ε value.
- **Location:** Slide "DBSCAN: Parameter Selection" (right side)
- **Suggested size:** 900x400px

---

## Section 05: Hierarchical Clustering

### 17. **05_agglomerative_vs_divisive_trees.png**
- **Type:** Two tree diagrams side-by-side
- **Description:**
  - Left (Agglomerative): Start with 5 individual nodes at bottom, arrows pointing up showing merging, single root at top
  - Right (Divisive): Start with 1 node at top, arrows pointing down showing splitting, 5 individual nodes at bottom
  Use different colors to distinguish the two approaches.
- **Location:** Slide "Hierarchical Clustering Overview" (right side)
- **Suggested size:** 700x500px

### 18. **05_agglomerative_steps_3frames.png**
- **Type:** 3-frame sequence
- **Description:**
  - Frame 1: 6 individual points (each its own cluster)
  - Frame 2: After 1-2 iterations showing some merges (4 clusters)
  - Frame 3: Multiple merges showing nested hierarchy (3 clusters with tree structure visible)
- **Location:** Slide "Agglomerative Hierarchical Clustering" (right side)
- **Suggested size:** 800x400px

### 19. **05_linkage_criteria_4methods.png**
- **Type:** 4 diagrams showing distance calculations
- **Description:** Show two clusters {A,B,C} and {D,E,F} with distance measurement lines:
  1. Single Linkage: Line between closest pair of points (A↔D)
  2. Complete Linkage: Line between farthest pair (C↔F)
  3. Average Linkage: Multiple lines showing average of all pairs
  4. Ward Linkage: Circles showing variance minimization around centroids
  Color-code distance lines differently (red/blue/green).
- **Location:** Slide "Linkage Criteria" (right side)
- **Suggested size:** 900x500px

### 20. **05_single_vs_complete_results.png**
- **Type:** Split-screen comparison (2 clusterings)
- **Description:** Same dataset clustered two ways:
  - Left (Single Linkage): Creates long, chain-like clusters (elongated shapes)
  - Right (Complete Linkage): Creates compact, spherical, more separated clusters
  Use different colors for clusters.
- **Location:** Slide "Single vs. Complete Linkage" (right side)
- **Suggested size:** 800x400px

### 21. **05_dendrogram_example.png**
- **Type:** Dendrogram diagram
- **Description:** Tree diagram with:
  - Horizontal axis: 8-10 point labels (A, B, C, D, E, F, etc.)
  - Vertical axis: Distance (0-10 scale)
  - Tree structure showing hierarchical merges at different heights
  - Clear merge points at different distance levels
- **Location:** Slide "Dendrograms" (right side)
- **Suggested size:** 700x500px

### 22. **05_dendrogram_with_cutlines.png**
- **Type:** Dendrogram with cut threshold lines
- **Description:** Same dendrogram as above but add 3-4 horizontal dotted lines at different heights:
  - Each line labeled with resulting cluster count
  - Example: Top line = "2 clusters", Middle line = "3 clusters", Lower line = "4 clusters"
  - Different colored regions to show resulting clusters at each cut level
- **Location:** Slide "Dendrogram Example" (right side)
- **Suggested size:** 700x500px

### 23. **05_hierarchical_vs_flat_comparison.png**
- **Type:** Split-screen comparison
- **Description:**
  - Left: Dendrogram showing tree structure and hierarchical relationships
  - Right: K-Means result showing flat clustering with fixed K=3 colored regions
  Same dataset used for both.
- **Location:** Slide "Hierarchical vs. Flat Clustering" (right side)
- **Suggested size:** 800x400px

---

## Section 06: Comparison & Summary

### 24. **06_clustering_methods_visual_summary.png**
- **Type:** 3x3 grid comparison
- **Description:** 3 example datasets (rows) × 3 methods (columns):
  - Rows: Blobs dataset, Moons dataset, Varying-density dataset
  - Columns: K-Means results, Hierarchical results, DBSCAN results
  Color-code clusters, mark noise points with × for DBSCAN
- **Location:** Slide "Clustering Methods Comparison" (bottom-right)
- **Suggested size:** 1000x700px

---

## Summary Statistics

- **Total images:** 24
- **Total files to create:** 24 PNG files
- **Folder location:** `fig/diagrams/`
- **Recommended format:** PNG with transparent background (where applicable)
- **Color scheme:** Consistent color palette for clusters (e.g., Red, Blue, Green, Yellow, Purple)
- **Suggested tools:** Python (matplotlib/seaborn), R (ggplot2), Adobe Illustrator, or any diagramming tool

---

## Naming Convention

All files follow the pattern: `NN_description.png` where:
- `NN` = section number (01-06)
- `description` = descriptive name (lowercase, underscores)
- Example: `03_kmeans_iterations_4frame.png`

---

## Next Steps

1. Create all 24 images using your preferred tool
2. Save to `fig/diagrams/` folder
3. The slide deck already has placeholders with `![...]` references ready
4. Export to PDF: `marp --theme themes/theme-mahidol.css DS08_clustering.md --pdf`
