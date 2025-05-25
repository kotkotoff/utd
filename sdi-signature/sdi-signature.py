import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SDISignature:
    def __init__(self, K=5, T=2, max_distance=0.15, strip_width=0.08, max_gap=0.15):
        self.K = K
        self.T = T
        self.max_distance = max_distance
        self.strip_width = strip_width  # Increased to 0.08 to capture more points
        self.max_gap = max_gap
        self.classifier = LogisticRegression(random_state=42, multi_class='ovr')
        self.scaler = StandardScaler()
        self.shapes = ["line", "circle", "triangle"]
        self.signatures = None
        self.all_segments = None
        self.all_strip_params = None
        self.labels = None

    def sort_points(self, points, shape_type):
        # Sort points along the shape to ensure correct ordering
        if shape_type == "line":
            # Sort by x-coordinate for a line
            indices = np.argsort(points[:, 0])
        elif shape_type == "circle":
            # Sort by angle relative to the center (0.5, 0.5) for a circle
            center = np.array([0.5, 0.5])
            angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
            indices = np.argsort(angles)
        else:  # triangle
            # Sort by angle relative to the centroid for a triangle
            centroid = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
            indices = np.argsort(angles)
        return points[indices], indices

    def detect_segments(self, points, shape_type):
        # Find neighbors to determine density
        nbrs = NearestNeighbors(n_neighbors=len(points)).fit(points)
        distances, indices = nbrs.kneighbors(points)

        segments = []
        strip_params = []

        if shape_type == "triangle":
            # Define triangle vertices (same as in generate_scene)
            v1 = np.array([0, 0])
            v2 = np.array([1, 0])
            v3 = np.array([0.5, np.sqrt(3) / 2])

            # Compute directions for each side
            dir1 = (v2 - v1) / np.linalg.norm(v2 - v1)  # Side 1: v1 to v2
            dir2 = (v3 - v2) / np.linalg.norm(v3 - v2)  # Side 2: v2 to v3
            dir3 = (v1 - v3) / np.linalg.norm(v1 - v3)  # Side 3: v3 to v1

            # Assign points to the nearest side and form strips
            for side_dir, side_start, side_end in [(dir1, v1, v2), (dir2, v2, v3), (dir3, v3, v1)]:
                strip_points = []
                for idx, point in enumerate(points):
                    # Compute distance to the side's axis
                    vec_to_start = point - side_start
                    projection = np.dot(vec_to_start, side_dir) * side_dir
                    perpendicular_vec = vec_to_start - projection
                    dist_to_axis = np.linalg.norm(perpendicular_vec)
                    # Check if the projection lies within the segment
                    proj_length = np.dot(vec_to_start, side_dir)
                    if 0 <= proj_length <= np.linalg.norm(side_end - side_start) and dist_to_axis <= self.strip_width:
                        strip_points.append((idx, proj_length))

                if len(strip_points) < self.T + 1:
                    continue

                # Sort points along the side
                strip_points.sort(key=lambda x: x[1])
                strip_indices = [idx for idx, _ in strip_points]
                strip_projections = [proj for _, proj in strip_points]

                # Split into continuous segments
                current_segment = [strip_indices[0]]
                current_projections = [strip_projections[0]]
                for idx, proj in zip(strip_indices[1:], strip_projections[1:]):
                    if abs(proj - current_projections[-1]) <= self.max_gap:
                        current_segment.append(idx)
                        current_projections.append(proj)
                    else:
                        if len(current_segment) >= self.T + 1:
                            segments.append(current_segment)
                            strip_params.append((side_start, side_dir, min(current_projections), max(current_projections)))
                        current_segment = [idx]
                        current_projections = [proj]
                if len(current_segment) >= self.T + 1:
                    segments.append(current_segment)
                    strip_params.append((side_start, side_dir, min(current_projections), max(current_projections)))
        else:
            # For lines and circles, use the existing method
            if shape_type == "line":
                # Compute a global axis for lines
                pca = PCA(n_components=1)
                pca.fit(points)
                global_direction = pca.components_[0]
                global_mean = np.mean(points, axis=0)

            for i in range(len(points)):
                xi = points[i]

                # Use global direction for lines, local for circles
                if shape_type == "line":
                    direction = global_direction
                    mean_point = global_mean
                else:  # Circle
                    neighbors = indices[i][1:]
                    neighbor_dists = distances[i][1:]
                    local_points_idx = [i]
                    local_points = [xi]
                    for j, dist in zip(neighbors, neighbor_dists):
                        if dist <= self.max_distance:
                            local_points_idx.append(j)
                            local_points.append(points[j])
                    local_points = np.array(local_points)
                    if len(local_points) < self.T + 1:
                        continue
                    pca = PCA(n_components=1)
                    pca.fit(local_points)
                    direction = pca.components_[0]
                    mean_point = np.mean(local_points, axis=0)

                # Collect points within the strip
                strip_points = []
                for idx, point in enumerate(points):
                    vec_to_point = point - mean_point
                    projection = np.dot(vec_to_point, direction) * direction
                    perpendicular_vec = vec_to_point - projection
                    dist_to_axis = np.linalg.norm(perpendicular_vec)
                    if dist_to_axis <= self.strip_width:
                        strip_points.append((idx, np.dot(point - mean_point, direction)))

                if len(strip_points) < self.T + 1:
                    continue

                # Sort points along the strip's axis
                strip_points.sort(key=lambda x: x[1])
                strip_indices = [idx for idx, _ in strip_points]
                strip_projections = [proj for _, proj in strip_points]

                # Split into continuous segments
                current_segment = [strip_indices[0]]
                current_projections = [strip_projections[0]]
                for idx, proj in zip(strip_indices[1:], strip_projections[1:]):
                    if abs(proj - current_projections[-1]) <= self.max_gap:
                        current_segment.append(idx)
                        current_projections.append(proj)
                    else:
                        if len(current_segment) >= self.T + 1:
                            segments.append(current_segment)
                            strip_params.append((mean_point, direction, min(current_projections), max(current_projections)))
                        current_segment = [idx]
                        current_projections = [proj]
                if len(current_segment) >= self.T + 1:
                    segments.append(current_segment)
                    strip_params.append((mean_point, direction, min(current_projections), max(current_projections)))

        return segments, strip_params

    def compute_extended_signature(self, points, segments, shape_type):
        # Compute extended signature with curvature
        if len(segments) == 0:
            return np.zeros(7)

        lengths = []
        directions = []
        curvatures = []

        for seg in segments:
            pts = points[seg]
            if len(pts) < 2:
                continue
            p0, p1 = pts[0], pts[-1]
            v = p1 - p0
            length = np.linalg.norm(v)
            if length == 0:
                continue
            dir_unit = v / length
            lengths.append(length)
            directions.append(dir_unit)

            # Compute curvature as the angle between consecutive segments
            if len(pts) >= 3:
                v1 = pts[1] - pts[0]
                v2 = pts[-1] - pts[-2]
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                if v1_norm > 0 and v2_norm > 0:
                    cos_angle = np.clip(np.dot(v1 / v1_norm, v2 / v2_norm), -1, 1)
                    curvature = np.abs(np.arccos(cos_angle))
                    curvatures.append(curvature)
                else:
                    curvatures.append(0)
            else:
                curvatures.append(0)

        if len(lengths) == 0:
            return np.zeros(7)

        lengths = np.array(lengths)
        directions = np.array(directions)
        curvatures = np.array(curvatures)

        mean_length = np.mean(lengths)
        length_std = np.std(lengths)
        max_length = np.max(lengths)
        avg_dir = np.mean(directions, axis=0)
        avg_dir /= (np.linalg.norm(avg_dir) + 1e-6)
        angle_var = np.mean([1 - np.dot(d, avg_dir) for d in directions])
        mean_curvature = np.mean(curvatures) if len(curvatures) > 0 else 0

        # Cluster directions to determine the number of direction clusters
        try:
            kmeans = KMeans(n_clusters=3, random_state=0).fit(directions)
            num_dir_clusters = len(set(kmeans.labels_))
        except:
            num_dir_clusters = 1

        return np.array([
            len(lengths),
            mean_length,
            length_std,
            max_length,
            angle_var,
            num_dir_clusters,
            mean_curvature
        ])

    def generate_scene(self, shape_type, num_points=50):
        # Generate a synthetic scene for the specified shape
        if shape_type == "line":
            t = np.linspace(0, 1, num_points)
            x = t + 0.05 * np.random.randn(num_points)
            y = 2 * t + 0.05 * np.random.randn(num_points)
        elif shape_type == "circle":
            theta = np.linspace(0, 2 * np.pi, num_points)
            x = 0.5 + 0.5 * np.cos(theta) + 0.05 * np.random.randn(num_points)
            y = 0.5 + 0.5 * np.sin(theta) + 0.05 * np.random.randn(num_points)
        else:  # triangle
            # Define vertices of an equilateral triangle
            v1 = np.array([0, 0])  # Vertex 1
            v2 = np.array([1, 0])  # Vertex 2
            v3 = np.array([0.5, np.sqrt(3) / 2])  # Vertex 3 (height = sqrt(3)/2 ≈ 0.866)
            # Distribute points evenly along the sides
            points_per_side = num_points // 3
            remaining_points = num_points - 3 * points_per_side
            # Side 1: v1 to v2
            t1 = np.linspace(0, 1, points_per_side + (1 if remaining_points > 0 else 0), endpoint=False)
            side1 = v1 + t1[:, np.newaxis] * (v2 - v1)
            if remaining_points > 0:
                remaining_points -= 1
            # Side 2: v2 to v3
            t2 = np.linspace(0, 1, points_per_side + (1 if remaining_points > 0 else 0), endpoint=False)
            side2 = v2 + t2[:, np.newaxis] * (v3 - v2)
            if remaining_points > 0:
                remaining_points -= 1
            # Side 3: v3 to v1
            t3 = np.linspace(0, 1, points_per_side + 1)  # Include the endpoint to close the triangle
            side3 = v3 + t3[:, np.newaxis] * (v1 - v3)
            # Combine points and add reduced noise
            points = np.vstack((side1, side2, side3))
            noise = 0.03 * np.random.randn(len(points), 2)  # Reduced noise to 0.03
            points += noise
            x = points[:, 0]
            y = points[:, 1]
        return np.vstack((x, y)).T

    def train(self, num_scenes_per_shape=5, num_points=50):
        # Train the model on synthetic data
        np.random.seed(42)
        scenes = []
        self.labels = []
        for shape in self.shapes:
            for _ in range(num_scenes_per_shape):
                scenes.append(self.generate_scene(shape, num_points))
                self.labels.append(self.shapes.index(shape))

        self.signatures = []
        self.all_segments = []
        self.all_strip_params = []
        for scene, label in zip(scenes, self.labels):
            shape_type = self.shapes[label]
            segments, strip_params = self.detect_segments(scene, shape_type)
            sig = self.compute_extended_signature(scene, segments, shape_type)
            self.signatures.append(sig)
            self.all_segments.append(segments)
            self.all_strip_params.append(strip_params)
        self.signatures = np.array(self.signatures)

        signatures_scaled = self.scaler.fit_transform(self.signatures)
        self.classifier.fit(signatures_scaled, self.labels)

    def predict(self, scenes, shape_types):
        # Predict classes for new scenes
        preds = []
        self.all_segments = []
        self.all_strip_params = []
        for scene, shape_type in zip(scenes, shape_types):
            segs, strip_params = self.detect_segments(scene, shape_type)
            sig = self.compute_extended_signature(scene, segs, shape_type)
            sig_scaled = self.scaler.transform(sig.reshape(1, -1))
            pred = self.classifier.predict(sig_scaled)[0]
            preds.append(pred)
            self.all_segments.append(segs)
            self.all_strip_params.append(strip_params)
        return preds

    def visualize(self, scenes, preds, labels, max_segments=None):
        # Visualize scenes with all segments and strips
        num_scenes_per_shape = len(scenes) // len(self.shapes)
        fig, axes = plt.subplots(len(self.shapes), num_scenes_per_shape, figsize=(15, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Different colors for strips

        for i, (scene, segments, strip_params, pred, label) in enumerate(zip(scenes, self.all_segments, self.all_strip_params, preds, labels)):
            row = i // num_scenes_per_shape
            col = i % num_scenes_per_shape
            ax = axes[row, col]
            ax.scatter(scene[:, 0], scene[:, 1], c='blue', s=10, label='Points')

            # Display all segments
            segments_to_show = segments if max_segments is None else segments[:max_segments]
            for seg in segments_to_show:
                if len(seg) >= 2:
                    pts = scene[seg]
                    ax.plot(pts[:, 0], pts[:, 1], 'r-', alpha=0.5)

            # Display all strips with different colors
            strip_params_to_show = strip_params if max_segments is None else strip_params[:max_segments]
            for j, (mean_point, direction, proj_min, proj_max) in enumerate(strip_params_to_show):
                p1 = mean_point + proj_min * direction
                p2 = mean_point + proj_max * direction
                normal = np.array([-direction[1], direction[0]])
                normal = normal / np.linalg.norm(normal) * self.strip_width
                length = np.linalg.norm(p2 - p1)
                angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
                color = colors[j % len(colors)]  # Cycle through colors
                rect = Rectangle(p1 - normal, length, 2 * self.strip_width, angle=angle, alpha=0.2, color=color)
                ax.add_patch(rect)

            ax.set_title(f"Scene {i+1}\nPred: {self.shapes[pred]}, True: {self.shapes[label]}")
            ax.legend()
        plt.tight_layout()
        plt.show()

    def evaluate(self, preds, labels):
        # Evaluate classification accuracy
        accuracy = np.mean(np.array(preds) == np.array(labels))
        print(f"Accuracy: {accuracy:.2f}")
        print("\nSignatures for analysis:")
        for i, (sig, label) in enumerate(zip(self.signatures, labels)):
            print(f"Scene {i+1} (True: {self.shapes[label]}): {sig}")

# Usage of the class
sdi_model = SDISignature(K=5, T=2, max_distance=0.18, strip_width=0.1, max_gap=0.19)

# Train the model
sdi_model.train(num_scenes_per_shape=5, num_points=51)  # Use 51 points to distribute evenly (17 per side)

# Generate test scenes
scenes = []
labels = []
shape_types = []
np.random.seed(42)
for shape in sdi_model.shapes:
    for _ in range(5):
        scenes.append(sdi_model.generate_scene(shape))
        labels.append(sdi_model.shapes.index(shape))
        shape_types.append(shape)

# Predict
preds = sdi_model.predict(scenes, shape_types)

# Visualize all segments and strips
sdi_model.visualize(scenes, preds, labels, max_segments=None)
sdi_model.evaluate(preds, labels)