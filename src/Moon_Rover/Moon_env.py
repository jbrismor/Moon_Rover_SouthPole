import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv
from heapq import heappush, heappop
import math
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

class LunarRover3DEnv(gym.Env):
    """Lunar Rover environment with persistent cold regions and proper movement"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                dem_path, 
                subregion_window=None, 
                render_mode="human", 
                max_slope_deg=25, 
                destination=None,
                smooth_sigma=None,
                desired_distance_m=20000,
                goal_radius_m=50,
                num_cold_regions=3,
                cold_region_scale=50,
                distance_reward_scale=0.08,
                max_num_steps=10000,
                radius_render=100,
                cold_region_locations=None,
                forward_speed = 10.0,
                step_penalty =  -0.001,
                cold_penalty = -100.0,
                slope_penalty = -10.0
                ):  # Modified parameter
        super().__init__()

        self.radius_render = radius_render
        self.render_mode = render_mode
        self.max_slope_deg = max_slope_deg
        self.goal_radius_m = goal_radius_m
        self.desired_distance_m = desired_distance_m
        self.distance_reward_scale = distance_reward_scale
        self.cold_region_scale = cold_region_scale
        self.path_found = None
        self.forward_speed = forward_speed
        self.step_penalty = step_penalty
        self.cold_penalty = cold_penalty
        self.slope_penalty = slope_penalty

        # Modified cold region handling
        self.cold_region_locations = cold_region_locations if cold_region_locations else []
        self.num_cold_regions = num_cold_regions  # Total regions (fixed + random)
        
        # Add these for path tracking
        self.record_path = False
        self.current_path = []
        self.agent_paths = []
        self.path_colors = {
        'Success': 'lime',
        'Crash': 'red',
        'Timeout': 'yellow'}
        self.agent_paths = []

        # --- Load the DEM ---
        with rasterio.open(dem_path) as src:
            # Get the resolution in meters
            self.x_res = src.transform[0]  # meters/pixel in x direction
            self.y_res = abs(src.transform[4])  # meters/pixel in y direction
            if subregion_window is not None:
                row_start, row_end, col_start, col_end = subregion_window
                window = rasterio.windows.Window(
                    col_start, row_start, 
                    col_end - col_start, row_end - row_start
                )
                self.dem = src.read(1, window=window)
            else:
                self.dem = src.read(1, masked=True)  # memory‐mapped

        # Optional smoothing:
        if smooth_sigma is not None:
            self.dem = gaussian_filter(self.dem, sigma=smooth_sigma)

        self.dem_shape = self.dem.shape
        self.dem_min = float(np.min(self.dem))
        self.dem_max = float(np.max(self.dem))

        self.max_distance_m = math.hypot(
            self.dem_shape[1] * self.x_res,
            self.dem_shape[0] * self.y_res)

        # Interpolator for continuous (row=y, col=x)
        y_coords = np.arange(self.dem_shape[0])
        x_coords = np.arange(self.dem_shape[1])
        self.dem_interp = RegularGridInterpolator((y_coords, x_coords), self.dem)

        # ---------- Gradients in “m/m” instead of “m/pixel” ----------
        raw_grad_x, raw_grad_y = np.gradient(self.dem)
        grad_x = raw_grad_x / self.x_res  
        grad_y = raw_grad_y / self.y_res
        self.grad_x = grad_x
        self.grad_y = grad_y

        # Define observation and action spaces
        obs_low = np.array([
            0.0, -1.0, 0.0, *[0.0]*16, 0.0, -1.0], dtype=np.float32)
        obs_high = np.array([
            1.0, 1.0, 90.0, *[90.0]*16, 1.0, 1.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0], dtype=np.float32),
            high=np.array([np.pi, 1.0], dtype=np.float32)
        )

        self.state = None
        self.max_steps = max_num_steps
        self.current_step = 0

        # Precompute grid for rendering
        X, Y = np.meshgrid(
            np.arange(self.dem_shape[1]) * self.x_res,
            np.arange(self.dem_shape[0]) * self.y_res
        )
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

        # Initialize cold regions (modified)
        self._init_cold_regions()
        self._precompute_safety_map()
        self._initialize_environment(destination)

        # Original print statements preserved
        print(f"DEM size: {self.dem.shape}")
        print(f"Min elevation: {self.dem_min}, Max elevation: {self.dem_max}")
        print(f"Spawn: {self.spawn}, Destination: {self.destination}")
        print(f"DEM resolution: x={self.x_res} m/px, y={self.y_res} m/px")
        print(f"Subregion size (meters): {self.dem_shape[0] * self.y_res / 1000} km x {self.dem_shape[1] * self.x_res / 1000} km")
        print(f"Cold regions: {len(self.cold_regions)} total ({len(self.cold_region_locations)} fixed, {len(self.cold_regions)-len(self.cold_region_locations)} random)")

        self.plotter = None

    def _precompute_safety_map(self):
        """Original safety map with slope checks"""
        grad_magnitude = np.sqrt(self.grad_x**2 + self.grad_y**2)
        self.safety_map = (np.abs(grad_magnitude) <= math.tan(math.radians(self.max_slope_deg)))
        self.safety_map &= ~self.cold_region_mask

    def _add_cold_region(self, mean_x, mean_y):
        """Add a single cold region to the mask"""
        xv, yv = np.meshgrid(np.arange(self.dem_shape[1]), 
                        np.arange(self.dem_shape[0]))
        gaussian = np.exp(-((xv - mean_x)**2 + (yv - mean_y)**2) / 
                    (2*self.cold_region_scale**2))
        self.cold_region_mask |= gaussian > 0.05
        self.cold_regions.append({'mean': (mean_x, mean_y)})

    def _init_cold_regions(self):
        """Modified cold region initialization"""
        self.cold_regions = []
        self.cold_region_mask = np.zeros(self.dem_shape, dtype=bool)
        
        # Add fixed regions
        for (x_m, y_m) in self.cold_region_locations:
            mean_x = x_m / self.x_res
            mean_y = y_m / self.y_res
            self._add_cold_region(mean_x, mean_y)
            print(f"Added fixed cold region at ({x_m}m, {y_m}m) -> pixel ({mean_x:.1f}, {mean_y:.1f})")

        # Add random regions to reach total count
        num_random = max(0, self.num_cold_regions - len(self.cold_region_locations))
        for _ in range(num_random):
            mean_x = np.random.uniform(0.2, 0.8) * self.dem_shape[1]
            mean_y = np.random.uniform(0.2, 0.8) * self.dem_shape[0]
            self._add_cold_region(mean_x, mean_y)
            print(f"Added random cold region at pixel ({mean_x:.1f}, {mean_y:.1f})")

    def _initialize_environment(self, destination):
        """Initialize spawn and destination points"""
        # Find safe spawn considering cold regions
        self.spawn = self.find_safe_spawn_point(
            (self.dem_shape[1]-1)/4, 
            (self.dem_shape[0]-1)*0.28)
        
        # Set destination
        self.destination = destination if destination else self.compute_reachable_destination_from_spawn(self.spawn)
        print(f"Spawn: {self.spawn}, Destination: {self.destination}")

    # -------------------------------------------------------------------------
    # Slope / Height Utilities
    # -------------------------------------------------------------------------

    def get_height(self, px, py):
        """Fast height lookup using interpolator"""
        return self.dem_interp((py, px))

    def get_slope(self, x, y, theta):
        """Calculate slope in heading direction using precomputed gradients"""
        xi = int(np.clip(x, 0, self.dem_shape[1]-1))
        yi = int(np.clip(y, 0, self.dem_shape[0]-1))
        
        # Get direction vector components (unit vector)
        dx = math.cos(theta)
        dy = math.sin(theta)
        dist = math.hypot(dx, dy)
        
        if dist < 1e-9:
            return 0.0
        
        # Normalize direction vector (though already unit length)
        dx /= dist
        dy /= dist
        
        # Dot product with gradient gives slope
        slope = dx * self.grad_x[yi, xi] + dy * self.grad_y[yi, xi]
        return abs(math.degrees(math.atan(slope)))
    
    def get_next_step_surrounding_inclinations(self, x, y, theta, distance):
        """Calculate slopes at positions `distance` meters ahead in 8 directions."""
        # Define 8 base directions as unit vectors (relative to EAST)
        base_directions = np.array([
            (1, 0),   # Front (aligned with heading)
            (1, 1),   # Front-right
            (0, 1),   # Right
            (-1, 1),  # Back-right
            (-1, 0),  # Back
            (-1, -1), # Back-left
            (0, -1),  # Left
            (1, -1)   # Front-left
        ], dtype=np.float32)
        
        # Normalize directions to unit vectors
        norms = np.linalg.norm(base_directions, axis=1, keepdims=True)
        base_directions = base_directions / norms
        
        # Rotate directions by agent's heading (theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        rotated_directions = np.dot(base_directions, rotation_matrix.T)
        
        # Scale directions by the given distance (in meters)
        directions_m = rotated_directions * distance
        
        slopes = []
        for dx_m, dy_m in directions_m:
            # Convert displacement to pixels
            dx_px = dx_m / self.x_res
            dy_px = dy_m / self.y_res
            new_x = x + dx_px
            new_y = y + dy_px
            
            # Clamp to DEM boundaries
            xi = int(np.clip(new_x, 0, self.dem_shape[1] - 1))
            yi = int(np.clip(new_y, 0, self.dem_shape[0] - 1))
            
            # Get gradient at new position
            grad_x = self.grad_x[yi, xi]
            grad_y = self.grad_y[yi, xi]
            
            # Calculate slope in the direction of movement (dx_m, dy_m)
            direction_magnitude = math.hypot(dx_m, dy_m)
            if direction_magnitude < 1e-9:
                slope_deg = 0.0
            else:
                dir_norm_x = dx_m / direction_magnitude
                dir_norm_y = dy_m / direction_magnitude
                slope = (dir_norm_x * grad_x) + (dir_norm_y * grad_y)
                slope_deg = abs(math.degrees(math.atan(slope)))
            
            slopes.append(slope_deg)
        
        return slopes

    
    def find_safe_spawn_point(self, initial_spawn_x, initial_spawn_y, search_radius=10):
        """
        Find a nearby safe spawn point if the initial one isn't suitable.
        Searches in expanding circles around the initial point.
        
        Args:
            initial_spawn_x, initial_spawn_y: Initial spawn coordinates
            search_radius: How far to search for a safe point
            
        Returns:
            (x, y) coordinates of a safe spawn point, or the initial point if none found
        """
        # First check if the initial point is already safe
        if (self.check_all_directions_slopes(initial_spawn_x, initial_spawn_y) and
            not self._in_cold_region(initial_spawn_x, initial_spawn_y)):
            return (initial_spawn_x, initial_spawn_y)
        
        # Search in expanding circles
        for radius in range(1, search_radius + 1):
            # Check points in a circle around the initial spawn
            for angle in range(0, 360, 10):  # Check every 10 degrees
                rad = math.radians(angle)
                x = initial_spawn_x + radius * math.cos(rad)
                y = initial_spawn_y + radius * math.sin(rad)
                
                # Ensure within DEM boundaries
                x = np.clip(x, 0, self.dem_shape[1] - 1)
                y = np.clip(y, 0, self.dem_shape[0] - 1)
                
                # Check if this point is safe
                if self.check_all_directions_slopes(x, y):
                    print(f"Found safe spawn at ({x}, {y}), {radius} pixels from initial spawn")
                    return (x, y)
        
        # If no safe point found, return the initial point and log a warning
        print(f"WARNING: Could not find safe spawn within {search_radius} pixels of initial spawn")
        return (initial_spawn_x, initial_spawn_y)
    
    def get_current_inclination(self, x, y, theta, forward_px=1.0):
        """Slope in degrees in the rover's heading direction"""
        # Simply use the heading direction for slope calculation
        return self.get_slope(x, y, theta)

    # def get_surrounding_inclinations(self, x, y):
    #     """Vectorized surrounding slope calculation"""
    #     xi = int(np.clip(x, 0, self.dem_shape[1]-1))
    #     yi = int(np.clip(y, 0, self.dem_shape[0]-1))
        
    #     # Precomputed directions in meters
    #     directions = np.array([
    #         (1, 0), (1, 1), (0, 1), (-1, 1),
    #         (-1, 0), (-1, -1), (0, -1), (1, -1)
    #     ]) * np.array([self.x_res, self.y_res])
        
    #     # Normalize directions
    #     norms = np.linalg.norm(directions, axis=1)
    #     directions_normalized = directions / norms[:, np.newaxis]
        
    #     # Calculate slopes
    #     slopes = (directions_normalized[:, 0] * self.grad_x[yi, xi] + 
    #             directions_normalized[:, 1] * self.grad_y[yi, xi])
    #     return np.abs(np.degrees(np.arctan(slopes))).tolist()

    def get_surrounding_inclinations(self, x, y, theta):
        """Calculate slopes in 8 directions relative to the agent's heading."""
        xi = int(np.clip(x, 0, self.dem_shape[1]-1))
        yi = int(np.clip(y, 0, self.dem_shape[0]-1))
        
        # Define 8 base directions as unit vectors (relative to EAST)
        base_directions = np.array([
            (1, 0),   # Front (aligned with heading)
            (1, 1),   # Front-right
            (0, 1),   # Right
            (-1, 1),  # Back-right
            (-1, 0),  # Back
            (-1, -1), # Back-left
            (0, -1),  # Left
            (1, -1)   # Front-left
        ], dtype=np.float32)
        
        # Normalize directions to unit vectors
        norms = np.linalg.norm(base_directions, axis=1, keepdims=True)
        base_directions = base_directions / norms
        
        # Rotate directions by agent's heading (theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        rotated_directions = np.dot(base_directions, rotation_matrix.T)
        
        # Convert directions to meters (using DEM resolution)
        directions_m = rotated_directions * np.array([self.x_res, self.y_res])
        
        # Calculate slopes
        slopes = []
        for dx_m, dy_m in directions_m:
            slope = (dx_m * self.grad_x[yi, xi] + dy_m * self.grad_y[yi, xi])
            slopes.append(abs(math.degrees(math.atan(slope))))
        
        return slopes

    # -------------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------------
    def get_observation(self):
        x, y, theta = self.state
        gx, gy = self.destination
        
        # Calculate relative distance and angle to destination
        delta_x_px = gx - x
        delta_y_px = gy - y
        
        # Convert delta to meters
        delta_x_m = delta_x_px * self.x_res
        delta_y_m = delta_y_px * self.y_res
        distance_m = math.hypot(delta_x_m, delta_y_m)
        normalized_distance = distance_m / self.max_distance_m  # Scale to [0, 1]
        
        # Angle to destination in global coordinates (radians)
        angle_to_dest = math.atan2(delta_y_m, delta_x_m)
        # Relative angle (difference between destination angle and rover's heading)
        relative_angle = (angle_to_dest - theta)  # In radians
        # Normalize to [-π, π], then scale to [-1, 1]
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
        relative_angle_normalized = relative_angle / math.pi
        
        # Current and surrounding inclinations
        current_incl = self.get_current_inclination(x, y, theta)
        current_surrounding = self.get_surrounding_inclinations(x, y, theta)
        next_step_surrounding = self.get_next_step_surrounding_inclinations(
        x, y, theta, self.forward_speed
    )
        
        # Cold region info (existing logic)
        near_dist, near_angle = self._get_nearest_cold_region_info(x, y, theta)
        scaled_dist = near_dist / self.max_distance_m
        scaled_angle = near_angle / np.pi
        
        return np.array([
            normalized_distance,
            relative_angle_normalized,
            current_incl,
            *current_surrounding,  # 8 values
            *next_step_surrounding,  # 8 values
            scaled_dist,
            scaled_angle
        ], dtype=np.float32)


    def _get_nearest_cold_region_info(self, x, y, theta):
        """Get distance and angle to nearest cold region with proper scaling"""
        min_dist = self.max_distance_m
        min_angle = 0.0
        
        for region in self.cold_regions:
            # Convert to meters
            rx = region['mean'][0] * self.x_res
            ry = region['mean'][1] * self.y_res
            px = x * self.x_res
            py = y * self.y_res
            
            # Calculate relative position
            dx = rx - px
            dy = ry - py
            dist = math.hypot(dx, dy)
            angle = math.atan2(dy, dx) - theta
            
            # Update minimum distance
            if dist < min_dist:
                min_dist = dist
                # Normalize angle to [-π, π]
                min_angle = (angle + np.pi) % (2 * np.pi) - np.pi
                
        return min_dist, min_angle

    def check_all_directions_slopes(self, x, y):
        """Ultra-fast slope safety check using precomputed safety map"""
        xi = int(np.clip(x, 0, self.dem_shape[1]-1))
        yi = int(np.clip(y, 0, self.dem_shape[0]-1))
        return self.safety_map[yi, xi]
    
    def is_transition_safe(self, x1, y1, x2, y2):
        """Check if all cells along the path are safe (slopes ≤ 25° in any direction)."""
        # Convert to grid indices
        x1_idx = int(np.clip(x1, 0, self.dem_shape[1]-1))
        y1_idx = int(np.clip(y1, 0, self.dem_shape[0]-1))
        x2_idx = int(np.clip(x2, 0, self.dem_shape[1]-1))
        y2_idx = int(np.clip(y2, 0, self.dem_shape[0]-1))
        
        # Check all cells along Bresenham’s line
        line = self._bresenham_line(x1_idx, y1_idx, x2_idx, y2_idx)
        for x, y in line:
            if not self.safety_map[y, x]:
                return False
        return True

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for accurate grid sampling"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        err = dx - dy
        line = []
        
        while True:
            line.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return line

    def _is_transition_safe_rc(self, row1, col1, row2, col2, step_px=0.5):
        """
        Slope check, but expects (row1, col1)).
        Internally we call self.is_transition_safe(x1, y1, x2, y2).
        """
        # Convert row,col -> x,y
        x1, y1 = float(col1), float(row1)
        x2, y2 = float(col2), float(row2)
        return self.is_transition_safe(x1, y1, x2, y2, step_px=step_px)
        
    def _reconstruct_path_rc(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def astar_path(self, start_rc, goal_rc):
        """
        Optimized A* pathfinding with precomputed safety map and meter-accurate checks
        Returns list of (row, col) coordinates from start_rc to goal_rc
        """
        if start_rc == goal_rc:
            return [start_rc] if self.safety_map[start_rc] else None

        # Precompute directions with costs (meters)
        neighbors = [
            (-1, 0, self.y_res),  # Up
            (1, 0, self.y_res),   # Down
            (0, -1, self.x_res),  # Left
            (0, 1, self.x_res),   # Right
            (-1, -1, math.hypot(self.x_res, self.y_res)),  # Diagonals
            (-1, 1, math.hypot(self.x_res, self.y_res)),
            (1, -1, math.hypot(self.x_res, self.y_res)),
            (1, 1, math.hypot(self.x_res, self.y_res)),
        ]

        # Initialize data structures
        open_heap = []
        came_from = {}
        g_score = np.full(self.dem_shape, np.inf)
        f_score = np.full(self.dem_shape, np.inf)
        
        # Convert start/goal to array indices
        start_y, start_x = start_rc
        goal_y, goal_x = goal_rc
        
        # Initialize starting node
        g_score[start_y, start_x] = 0
        f_score[start_y, start_x] = self._heuristic(start_rc, goal_rc)
        heappush(open_heap, (f_score[start_y, start_x], 0, start_y, start_x))

        # Precompute safety check thresholds
        max_slope_tan = math.tan(math.radians(self.max_slope_deg))
        safe_cells = np.where(self.safety_map)
        
        while open_heap:
            current_f, _, cy, cx = heappop(open_heap)
            
            # Early exit if goal reached
            if (cy, cx) == (goal_y, goal_x):
                return self._reconstruct_path_rc(came_from, (cy, cx))
            
            # Skip processed nodes
            if current_f > f_score[cy, cx]:
                continue
                
            for dy, dx, step_cost in neighbors:
                ny = cy + dy
                nx = cx + dx
                
                # Boundary check
                if not (0 <= ny < self.dem_shape[0] and 0 <= nx < self.dem_shape[1]):
                    continue
                    
                # Fast safety check
                if not self.safety_map[ny, nx]:
                    continue
                    
                # Detailed transition check in meters
                if not self._safe_transition_meters((cy, cx), (ny, nx)):
                    continue
                    
                # Calculate tentative g-score
                tentative_g = g_score[cy, cx] + step_cost
                
                if tentative_g < g_score[ny, nx]:
                    came_from[(ny, nx)] = (cy, cx)
                    g_score[ny, nx] = tentative_g
                    f_score[ny, nx] = tentative_g + self._heuristic((ny, nx), goal_rc)
                    heappush(open_heap, (f_score[ny, nx], id((ny, nx)), ny, nx))

        return None  # No path found
    
    def _safe_transition_meters(self, from_rc, to_rc):
        """Check transition safety using Bresenham's line algorithm in meters"""
        line = self._bresenham_line(from_rc[1], from_rc[0], to_rc[1], to_rc[0])
        for x, y in line:
            if not self.safety_map[y, x]:
                return False
        return True
    
    def _heuristic(self, a, b):
        """Meter-accurate octile distance heuristic"""
        dx = abs(a[1] - b[1]) * self.x_res
        dy = abs(a[0] - b[0]) * self.y_res
        return max(dx, dy) + (math.sqrt(2)-1) * min(dx, dy)

    def _reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append((current[1], current[0]))
            current = came_from[current]
        path.reverse()
        return path

    def check_path_possible(self, spawn, candidate_goal):
        """
        Return the A* path from spawn to candidate_goal if one exists,
        otherwise return None.
        """
        start_rc = (int(round(spawn[1])), int(round(spawn[0])))
        goal_rc = (int(round(candidate_goal[1])), int(round(candidate_goal[0])))
        # Return the actual path (or None)
        return self.astar_path(start_rc, goal_rc)

    def compute_reachable_destination_from_spawn(self, spawn):
        """More efficient reachable destination computation"""
        desired_distance_px_x = self.desired_distance_m / self.x_res
        desired_distance_px_y = self.desired_distance_m / self.y_res
        
        # Try fewer angles first
        angles = np.linspace(0, 360, 12, endpoint=False)  # 30-degree intervals
        
        for angle_deg in angles:
            rad = math.radians(angle_deg)
            cx = spawn[0] + desired_distance_px_x * math.cos(rad)
            cy = spawn[1] + desired_distance_px_y * math.sin(rad)
            cx = np.clip(cx, 0, self.dem_shape[1] - 1)
            cy = np.clip(cy, 0, self.dem_shape[0] - 1)
            candidate = (cx, cy)
            
            # Check if slope is safe at destination
            if not self.check_all_directions_slopes(cx, cy):
                continue
                
            path = self.check_path_possible(spawn, candidate)
            if path is not None:
                self.path_found = path
                return candidate
        
        # If no path found with 30-degree intervals, try more angles
        angles = np.linspace(0, 360, 36, endpoint=False)  # 10-degree intervals
        
        def check_angle(angle_deg):
            rad = math.radians(angle_deg)
            cx = spawn[0] + desired_distance_px_x * math.cos(rad)
            cy = spawn[1] + desired_distance_px_y * math.sin(rad)
            cx = np.clip(cx, 0, self.dem_shape[1] - 1)
            cy = np.clip(cy, 0, self.dem_shape[0] - 1)
            candidate = (cx, cy)
            
            # Check if slope is safe at destination
            if not self.check_all_directions_slopes(cx, cy):
                return None
                
            path = self.check_path_possible(spawn, candidate)
            return (candidate, path) if path is not None else None
        
        # Check angles in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(check_angle, angles))
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            # Get first valid result
            candidate, path = valid_results[0]
            self.path_found = path
            return candidate
        
        # If no path found, try closer destinations
        for distance_factor in [0.75, 0.5, 0.25]:
            shorter_distance_px_x = desired_distance_px_x * distance_factor
            shorter_distance_px_y = desired_distance_px_y * distance_factor
            
            for angle_deg in angles:
                rad = math.radians(angle_deg)
                cx = spawn[0] + shorter_distance_px_x * math.cos(rad)
                cy = spawn[1] + shorter_distance_px_y * math.sin(rad)
                cx = np.clip(cx, 0, self.dem_shape[1] - 1)
                cy = np.clip(cy, 0, self.dem_shape[0] - 1)
                candidate = (cx, cy)
                
                path = self.check_path_possible(spawn, candidate)
                if path is not None:
                    self.path_found = path
                    return candidate
        
        return spawn  # Fallback to spawn if no path found

    # -------------------------------------------------------------------------
    # Gym Interface
    # -------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Get the initial spawn coordinates
        initial_x, initial_y = self.spawn
        
        # Verify the spawn point is safe, find an alternative if not
        safe_x, safe_y = self.find_safe_spawn_point(initial_x, initial_y)
        
        # Update the spawn point if it changed
        if (safe_x, safe_y) != (initial_x, initial_y):
            self.spawn = (safe_x, safe_y)
            
            # Optionally recalculate the destination if spawn changed significantly
            spawn_changed_significantly = np.linalg.norm(np.array([safe_x - initial_x, safe_y - initial_y])) > 5
            if spawn_changed_significantly and self.destination is not None:
                print("Spawn changed significantly, recalculating destination")
                self.destination = self.compute_reachable_destination_from_spawn(self.spawn)
        
        # State = [x, y, theta]
        self.state = np.array([safe_x, safe_y, 0.0], dtype=np.float32)

        self.goal = self.destination

        obs = self.get_observation()

        # Initialize path recording
        if options and options.get('record_path', False):
            self.record_path = True
            self.current_path = []
        else:
            self.record_path = False
            
        return obs, {}

    # def step(self, action):

    #     forward_speed = self.forward_speed
    #     step_penalty = self.step_penalty
    #     cold_penalty = self.cold_penalty
    #     slope_penalty = self.slope_penalty 
    
    #     prev_state = np.copy(self.state)
    #     turn, forward = action
    #     x, y, theta = self.state
        
    #     # 1. Scale forward action to realistic speed (10 meters per step)
    #     forward_meters = forward * forward_speed # Now 0-10 meters per step
    #     delta_x = (forward_meters * math.cos(theta)) / self.x_res  # Convert meters → pixels
    #     delta_y = (forward_meters * math.sin(theta)) / self.y_res
        
    #     new_x = x + delta_x
    #     new_y = y + delta_y
    #     new_x = np.clip(new_x, 0, self.dem_shape[1]-1)
    #     new_y = np.clip(new_y, 0, self.dem_shape[0]-1)
    #     new_theta = (theta + turn) % (2 * np.pi)
        
    #     # 2. Validate entire movement path
    #     if not self.is_transition_safe(x, y, new_x, new_y):
    #         # 3. Crash type detection (slope or cold region)
    #         path = self._bresenham_line(int(x), int(y), int(new_x), int(new_y))
    #         cold_crash = any(
    #             self._in_cold_region(cx, cy) 
    #             for (cx, cy) in path
    #         )
    #         penalty = cold_penalty if cold_crash else slope_penalty
    #         print(f"Crash: {'Cold region' if cold_crash else 'Steep slope'}")
    #         return self.get_observation(), penalty, True, False, {}
        
    #     # 4. Update state only if path is safe
    #     self.state = np.array([new_x, new_y, new_theta], dtype=np.float32)
    #     self.current_step += 1
        
    #     # 5. Reward calculation
    #     reward = step_penalty  # Step penalty
    #     reward += self._calculate_distance_reward(prev_state)  # Distance-based reward
        
    #     # 6. Goal check
    #     done, goal_reward = self._check_goal_condition()
    #     reward += goal_reward
        
    #     # 7. Step limit check
    #     if self.current_step >= self.max_steps:
    #         done = True
    #         print("Max steps reached!")
        
    #     return self.get_observation(), reward, done, False, {}

    def step(self, action):
        prev_state = np.copy(self.state)
        turn, forward = action
        x, y, theta = self.state
        
        # 1. Scale forward action to realistic speed (20 meters per step)
        forward_meters = forward * self.forward_speed
        delta_x = (forward_meters * math.cos(theta)) / self.x_res  # Convert meters → pixels
        delta_y = (forward_meters * math.sin(theta)) / self.y_res
        
        new_x = x + delta_x
        new_y = y + delta_y
        new_x = np.clip(new_x, 0, self.dem_shape[1]-1)
        new_y = np.clip(new_y, 0, self.dem_shape[0]-1)
        new_theta = (theta + turn) % (2 * np.pi)
        
        # 2. Validate entire movement path
        if not self.is_transition_safe(x, y, new_x, new_y):
            # Crash detection
            path = self._bresenham_line(int(x), int(y), int(new_x), int(new_y))
            cold_crash = any(self._in_cold_region(cx, cy) for (cx, cy) in path)
            penalty = self.cold_penalty if cold_crash else self.slope_penalty
            print(f"Crash: {'Cold region' if cold_crash else 'Steep slope'}")
            return self.get_observation(), penalty, True, False, {}
        
        # 3. Update state if safe
        self.state = np.array([new_x, new_y, new_theta], dtype=np.float32)
        self.current_step += 1
        
        # 4. Record path point (in meters)
        if self.record_path:
            x_m = new_x * self.x_res
            y_m = new_y * self.y_res
            z_m = self.get_height(new_x, new_y)
            self.current_path.append([x_m, y_m, z_m + 10])  # Store 10m above surface
        
        # 5. Calculate reward
        reward = self.step_penalty  # Base penalty per step
        reward += self._calculate_distance_reward(prev_state)  # Distance improvement
        
        # 6. Check termination conditions
        done, goal_reward = self._check_goal_condition()
        reward += goal_reward
        
        if self.current_step >= self.max_steps:
            done = True
            print("Max steps reached!")
        
        return self.get_observation(), reward, done, False, {}

    def _in_cold_region(self, x, y):
        """Fast cold region check"""
        xi = int(round(np.clip(x, 0, self.dem_shape[1]-1)))
        yi = int(round(np.clip(y, 0, self.dem_shape[0]-1)))
        return self.cold_region_mask[yi, xi]

    def _calculate_distance_reward(self, prev_state):
        """Calculate scaled reward based on distance improvement"""
        gx, gy = self.destination
        x_prev, y_prev, _ = prev_state
        x_new, y_new, _ = self.state
        
        prev_dist = math.hypot((x_prev - gx)*self.x_res, (y_prev - gy)*self.y_res)
        new_dist = math.hypot((x_new - gx)*self.x_res, (y_new - gy)*self.y_res)
        return (prev_dist - new_dist) * self.distance_reward_scale
    
    def _check_goal_condition(self):
        """Check if rover reached goal"""
        gx, gy = self.destination
        x, y, _ = self.state
        dist = math.hypot((x - gx)*self.x_res, (y - gy)*self.y_res)
        if dist < self.goal_radius_m:
            print(f"GOAL REACHED! dist={dist:.2f} < {self.goal_radius_m} => +500 reward")
            return True, 500.0
        else:
            return False, 0.0

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def _add_astar_path_mesh(self, path):
        """
        Draws the A* path as a polyline above the terrain.
        path is a list of (row, col) = (y, x).
        """
        pts = []
        for (ry, cx) in path:
            x_m = cx * self.x_res
            y_m = ry * self.y_res
            z_m = self.dem[ry, cx] + 10  # slightly above
            pts.append([x_m, y_m, z_m])
        pts = np.array(pts, dtype=np.float32)

        poly = pv.PolyData()
        poly.points = pts
        lines = []
        for i in range(len(pts)-1):
            lines.append(2)
            lines.append(i)
            lines.append(i+1)
        poly.lines = np.array(lines)
        self.plotter.add_mesh(poly, color='yellow', line_width=5, render_lines_as_tubes=True)

    def _add_path_to_plotter(self, path_data):
        """Helper to add a path to the PyVista plotter"""
        pts = np.array(path_data['points'], dtype=np.float32)
        if len(pts) < 2:
            return

        poly = pv.PolyData()
        poly.points = pts
        lines = []
        for i in range(len(pts)-1):
            lines.append(2)
            lines.append(i)
            lines.append(i+1)
        poly.lines = np.array(lines)
        self.plotter.add_mesh(poly, color=path_data['color'], line_width=8, render_lines_as_tubes=True)

    # def render(self, mode='human', show_path=False, save_frames=False):
    #     """
    #     Renders the environment using PyVista.
    #     If 'show_path=True' and self.path_found is not None, draws the path in 3D.
    #     Cold regions are rendered in black.
    #     """
    #     if self.plotter is None:
    #         self.plotter = pv.Plotter(window_size=(1024, 768))
    #     else:
    #         self.plotter.clear()

    #     radius_render = self.radius_render

    #     # Create structured grid for main DEM
    #     grid = pv.StructuredGrid(self.X, self.Y, self.dem)
    #     self.plotter.add_mesh(grid, cmap="Blues_r", show_scalar_bar=True)
        
    #     # Add cold regions in black
    #     if hasattr(self, 'cold_region_mask'):
    #         cold_z = np.where(self.cold_region_mask, self.dem, np.nan)
    #         cold_grid = pv.StructuredGrid(self.X, self.Y, cold_z)
    #         self.plotter.add_mesh(cold_grid, color='black', opacity=0.7)

    #     # Rover
    #     x_px, y_px, theta = self.state
    #     x_m = x_px * self.x_res
    #     y_m = y_px * self.y_res
    #     z_m = self.get_height(x_px, y_px)

    #     # Create a sphere with a specified center and radius
    #     rover = pv.Sphere(
    #         center=(x_m, y_m, z_m + 10),  # Adjust z offset as needed
    #         radius=radius_render                # Change radius value to suit your rover size
    #     )

    #     # Note: A sphere is symmetric so a rotation based on theta is not necessary.
    #     self.plotter.add_mesh(rover, color="red")

    #     # Goal
    #     gx_px, gy_px = self.destination
    #     gx_m = gx_px * self.x_res
    #     gy_m = gy_px * self.y_res
    #     gz_m = self.get_height(gx_px, gy_px)
    #     goal_marker = pv.Sphere(radius=radius_render, center=(gx_m, gy_m, gz_m))
    #     self.plotter.add_mesh(goal_marker, color="green")

    #     # Optionally draw the A* path
    #     if show_path and self.path_found is not None:
    #         self._add_astar_path_mesh(self.path_found)

    #     # Camera setup remains unchanged
    #     offset = 150000
    #     self.plotter.camera_position = [
    #         (x_m + offset, y_m + offset, z_m + offset),
    #         (x_m, y_m, z_m),
    #         (0, 0, 1)
    #     ]

    #     self.plotter.show_grid(
    #         xtitle="X (meters)",
    #         ytitle="Y (meters)",
    #         ztitle="Elevation (meters)",
    #     )

    #     # Add agent paths
    #     for path_data in self.agent_paths:
    #         self._add_path_to_plotter(path_data)

    #     if save_frames:
    #         return self.plotter.screenshot(return_img=True)
    #     else:
    #         self.plotter.show()

    def set_plotter(self, plotter):
        """Let an external script attach its own PyVista plotter."""
        self.plotter = plotter

    def render(self, mode='human', show_path=False, save_frames=False):
        """
        Renders the environment using PyVista with the original "Blues_r" style
        but now also supports multiple path renderings from self.agent_paths.
        Cold regions are rendered in black.
        """
        if self.plotter is None:
            self.plotter = pv.Plotter(window_size=(1024, 768))
        else:
            self.plotter.clear()

        # Use the same radius as before
        radius_render = self.radius_render

        # 1) Create the structured grid for the main DEM
        grid = pv.StructuredGrid(self.X, self.Y, self.dem)
        self.plotter.add_mesh(
            grid,
            cmap="Blues_r",             # Matches original style
            show_scalar_bar=True,
            clim=(self.dem_min, self.dem_max),
        )

        # 2) Cold regions in black
        if hasattr(self, 'cold_region_mask'):
            cold_z = np.where(self.cold_region_mask, self.dem, np.nan)
            cold_grid = pv.StructuredGrid(self.X, self.Y, cold_z)
            self.plotter.add_mesh(cold_grid, color='black', opacity=0.7)

        # 3) Rover location
        x_px, y_px, theta = self.state
        x_m = x_px * self.x_res
        y_m = y_px * self.y_res
        z_m = self.get_height(x_px, y_px)
        rover = pv.Sphere(center=(x_m, y_m, z_m + 10), radius=radius_render)
        self.plotter.add_mesh(rover, color="red")

        # 4) Goal location
        gx_px, gy_px = self.destination
        gx_m = gx_px * self.x_res
        gy_m = gy_px * self.y_res
        gz_m = self.get_height(gx_px, gy_px)
        goal_marker = pv.Sphere(radius=radius_render, center=(gx_m, gy_m, gz_m))
        self.plotter.add_mesh(goal_marker, color="green")

        # 5) Optionally draw a single “A*” path, if you still want that
        if show_path and self.path_found is not None:
            self._add_astar_path_mesh(self.path_found)

        # 6) Draw any additional agent paths in self.agent_paths
        for path_data in self.agent_paths:
            self._add_path_to_plotter(path_data)

        # --- OPTIONAL: Label each path if desired ---
        # for idx, path_data in enumerate(self.agent_paths):
        #     text = f"Try {idx+1}: {path_data['outcome']}"
        #     self.plotter.add_text(
        #         text,
        #         position=(0.05, 0.95 - idx*0.04),
        #         font_size=12,
        #         color=path_data['color'],
        #         font='arial'
        #     )

        # 7) Large camera offset like before
        offset = 150000
        self.plotter.camera_position = [
            (x_m + offset, y_m + offset, z_m + offset),
            (x_m, y_m, z_m),
            (0, 0, 1)
        ]

        # 8) Axis labels
        self.plotter.show_grid(
            xtitle="X (meters)",
            ytitle="Y (meters)",
            ztitle="Elevation (meters)",
        )

        # 9) Show or return screenshot if saving
        if save_frames:
            return self.plotter.screenshot(return_img=True)
        else:
            self.plotter.show()

    def close(self):
        """Optional close method if you want to do something special."""
        pass

# -------------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------------
    # # Optionally run A* from spawn -> destination
    # spawn_rc = (int(round(env.spawn[1])), int(round(env.spawn[0])))
    # goal_rc = (int(round(env.destination[1])), int(round(env.destination[0])))
    # env.path_found = env.astar_path(spawn_rc, goal_rc)

if __name__ == "__main__":
    dem_file_path = "src/map/LDEM_80S_20MPP_ADJ.tiff"
    subregion_window = (6000, 7000, 6000, 7000)
    
    # Create environment with cold region at (29985m, 995m)
    env = LunarRover3DEnv(
        dem_file_path,
        radius_render=10,
        subregion_window = subregion_window,
        desired_distance_m=1000,
        cold_region_scale=10,
        # cold_region_locations=[(29985, 10000)] , # Now accepts meter coordinates
        num_cold_regions=1
    )
    
    obs, _ = env.reset()
    env.render(show_path=True)