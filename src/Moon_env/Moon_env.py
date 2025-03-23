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
    """
    Checkpoint 1 environment for the Lunar Rover project.:
      1) Uses a high-resolution DEM (digital elevation model) as the terrain.
      2) Allows continuous turning + forward movement actions.
      3) Computes terrain slopes in meters (true physical distance).
      4) Terminates (crash) if local slope > ±25°.
      5) Returns an observation with:
           [ x, y, x_dest, y_dest, current_inclination,
             surrounding_inclinations (8 values) ] 
         in degrees, plus x,y in pixel coordinates. 
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                dem_path, 
                subregion_window=None, 
                render_mode="human", 
                max_slope_deg=25, 
                destination=None,
                smooth_sigma=None,
                desired_distance_m=70000,
                goal_radius_m=50,
                num_cold_regions=3,
                cold_region_scale=20,
                distance_reward_scale=0.02):
        super().__init__()

        self.render_mode = render_mode
        self.max_slope_deg = max_slope_deg
        self.goal_radius_m = goal_radius_m
        self.desired_distance_m = desired_distance_m
        self.distance_reward_scale = distance_reward_scale
        self.num_cold_regions = num_cold_regions
        self.cold_region_scale = cold_region_scale
        self.path_found = None

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

        # ---------- NEW CODE for gradients in “m/m” instead of “m/pixel” ----------
        # 1) get the raw gradient in pixel units and then to meters
        raw_grad_x, raw_grad_y = np.gradient(self.dem)
        grad_x = raw_grad_x / self.x_res  
        grad_y = raw_grad_y / self.y_res
        self.grad_x = grad_x
        self.grad_y = grad_y

        # --------------------------------------------------------------------------

        # Define the observation and action spaces
        obs_low = np.array([
            0.0, 0.0,              # x, y (pixel coordinates)
            0.0, 0.0,              # x_dest, y_dest (pixels)
            -90.0,                 # current_inclination (degrees)
            *[-90.0]*8,            # 8 surrounding inclinations
            0.0,                   # scaled_distance [0,1]
            -1.0                   # scaled_angle [-1,1]
        ], dtype=np.float32)

        obs_high = np.array([
            self.dem_shape[1]-1, self.dem_shape[0]-1,  # x, y max
            self.dem_shape[1]-1, self.dem_shape[0]-1,  # dest_x, dest_y max
            90.0,                  # current_inclination max
            *[90.0]*8,             # surrounding inclinations max
            1.0,                   # scaled_distance max
            1.0                    # scaled_angle max
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0], dtype=np.float32),
            high=np.array([np.pi, 1.0], dtype=np.float32)
        )

        self.state = None  # [x, y, theta]
        self.max_steps = 10000
        self.current_step = 0

        # Precompute grid in meters for rendering
        X, Y = np.meshgrid(
            np.arange(self.dem_shape[1]) * self.x_res,
            np.arange(self.dem_shape[0]) * self.y_res
        )
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

        # Initialize cold regions
        self._init_cold_regions()

        # Precompute safety map
        self.max_slope_tan = math.tan(math.radians(self.max_slope_deg))
        self._precompute_safety_map() 
        self._initialize_environment(destination)

        print(f"DEM size: {self.dem.shape}")
        print(f"Min elevation: {self.dem_min}, Max elevation: {self.dem_max}")

        # Spawn logic
        center_x = (self.dem_shape[1] - 1) / 4.0
        center_y = (self.dem_shape[0] - 1) * 0.280
        self.spawn = (center_x, center_y)

        # Destination
        if destination is None:
            self.destination = self.compute_reachable_destination_from_spawn(self.spawn)
        else:
            self.destination = destination
            start_rc = (int(round(self.spawn[1])), int(round(self.spawn[0])))
            goal_rc = (int(round(self.destination[1])), int(round(self.destination[0])))
            self.path_found = self.astar_path(start_rc, goal_rc)

        print(f"Spawn: {self.spawn}, Destination: {self.destination}")
        print(f"DEM resolution: x={self.x_res} m/px, y={self.y_res} m/px")
        print(f"Subregion size (meters): {self.dem_shape[0] * self.y_res / 1000} km x {self.dem_shape[1] * self.x_res / 1000} km")

        self.plotter = None

    def _precompute_safety_map(self):
        """Precompute safety map for rapid slope checks"""
        # Calculate gradient magnitude
        grad_magnitude = np.sqrt(self.grad_x**2 + self.grad_y**2)
        # Create safety map (True = safe)
        self.safety_map = (np.abs(grad_magnitude) <= self.max_slope_tan)
        self.safety_map &= ~self.cold_region_mask

    def _init_cold_regions(self):
        """Properly initialized cold regions"""
        self.cold_regions = []
        self.cold_region_mask = np.zeros(self.dem_shape, dtype=bool)
        
        for _ in range(self.num_cold_regions):
            mean_x = np.random.uniform(0.2, 0.8) * self.dem_shape[1]
            mean_y = np.random.uniform(0.2, 0.8) * self.dem_shape[0]
            sigma = self.cold_region_scale
            xv, yv = np.meshgrid(np.arange(self.dem_shape[1]), 
                            np.arange(self.dem_shape[0]))
            
            # Create 2D Gaussian and add to mask
            gaussian = np.exp(-((xv-mean_x)**2 + (yv-mean_y)**2) / (2*sigma**2))
            self.cold_region_mask |= gaussian > 0.05  # Threshold at 5% intensity

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
        return math.degrees(math.atan(slope))

    
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

    def get_surrounding_inclinations(self, x, y):
        """Vectorized surrounding slope calculation"""
        xi = int(np.clip(x, 0, self.dem_shape[1]-1))
        yi = int(np.clip(y, 0, self.dem_shape[0]-1))
        
        # Precomputed directions in meters
        directions = np.array([
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]) * np.array([self.x_res, self.y_res])
        
        # Normalize directions
        norms = np.linalg.norm(directions, axis=1)
        directions_normalized = directions / norms[:, np.newaxis]
        
        # Calculate slopes
        slopes = (directions_normalized[:, 0] * self.grad_x[yi, xi] + 
                directions_normalized[:, 1] * self.grad_y[yi, xi])
        return np.degrees(np.arctan(slopes)).tolist()

    # -------------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------------
    def get_observation(self):
        """Enhanced observation with scaled cold region information"""
        x, y, theta = self.state
        gx, gy = self.destination
        
        # Terrain information
        current_incl = self.get_current_inclination(x, y, theta)
        surrounding_incl = self.get_surrounding_inclinations(x, y)
        
        # Cold region information (scaled)
        near_dist, near_angle = self._get_nearest_cold_region_info(x, y, theta)
        scaled_dist = near_dist / self.max_distance_m  # Scale to [0,1]
        scaled_angle = near_angle / np.pi  # Scale to [-1,1]

        return np.array([
            x, y, gx, gy, current_incl,
            *surrounding_incl,
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
    
    def is_transition_safe(self, x1, y1, x2, y2, step_px=0.5):
        """Optimized path safety check"""
        # Convert positions to array indices
        x1_idx = int(np.clip(x1, 0, self.dem_shape[1]-1))
        y1_idx = int(np.clip(y1, 0, self.dem_shape[0]-1))
        x2_idx = int(np.clip(x2, 0, self.dem_shape[1]-1))
        y2_idx = int(np.clip(y2, 0, self.dem_shape[0]-1))
        
        # Check endpoints
        if not (self.safety_map[y1_idx, x1_idx] and self.safety_map[y2_idx, x2_idx]):
            return False
        
        # Check intermediate points using Bresenham's line algorithm
        line_points = list(zip(*self._bresenham_line(x1_idx, y1_idx, x2_idx, y2_idx)))
        for x, y in line_points:
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
        Same slope check, but expects (row1, col1) instead of (x1, y1).
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
        return obs, {}

    def step(self, action):
        prev_state = np.copy(self.state)
        turn, forward = action
        x, y, theta = self.state
        
        # Update position
        theta = (theta + turn) % (2 * np.pi)
        new_x = x + forward * math.cos(theta)
        new_y = y + forward * math.sin(theta)
        new_x = np.clip(new_x, 0, self.dem_shape[1]-1)
        new_y = np.clip(new_y, 0, self.dem_shape[0]-1)

        # Check for cold regions
        if self._in_cold_region(new_x, new_y):
            return self.get_observation(), -100.0, True, False, {}

        # Check slope safety
        if not self.check_all_directions_slopes(new_x, new_y):
            return self.get_observation(), -50.0, True, False, {}

        # Update state
        self.state = np.array([new_x, new_y, theta], dtype=np.float32)
        self.current_step += 1

        # Calculate rewards
        reward = -0.1  # Step penalty
        reward += self._calculate_distance_reward(prev_state)
        
        # Check goal condition
        done, goal_reward = self._check_goal_condition()
        reward += goal_reward

        # Check step limit
        if self.current_step >= self.max_steps:
            done = True

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
        return (dist < self.goal_radius_m), 100.0 if (dist < self.goal_radius_m) else 0.0

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
        self.plotter.add_mesh(poly, color='yellow', line_width=5)

    def render(self, mode='human', show_path=False, save_frames=False):
        """
        Renders the environment using PyVista. 
        If 'show_path=True' and self.path_found is not None, draws the path in 3D.
        Cold regions are rendered in black.
        """
        if self.plotter is None:
            self.plotter = pv.Plotter(window_size=(1024, 768))
        else:
            self.plotter.clear()

        # Create structured grid for main DEM
        grid = pv.StructuredGrid(self.X, self.Y, self.dem)
        self.plotter.add_mesh(grid, cmap="Blues_r", show_scalar_bar=True)
        
        # Add cold regions in black
        if hasattr(self, 'cold_region_mask'):
            cold_z = np.where(self.cold_region_mask, self.dem, np.nan)
            cold_grid = pv.StructuredGrid(self.X, self.Y, cold_z)
            self.plotter.add_mesh(cold_grid, color='black', opacity=0.7)

        # Rover
        x_px, y_px, theta = self.state
        x_m = x_px * self.x_res
        y_m = y_px * self.y_res
        z_m = self.get_height(x_px, y_px)

        rover = pv.Plane(
            center=(x_m, y_m, z_m + 5),
            direction=(0,0,1),
            i_size=1000, 
            j_size=1000
        )
        rover.rotate_z(np.degrees(theta), inplace=True)
        self.plotter.add_mesh(rover, color="red")

        # Goal
        gx_px, gy_px = self.destination
        gx_m = gx_px * self.x_res
        gy_m = gy_px * self.y_res
        gz_m = self.get_height(gx_px, gy_px)
        goal_marker = pv.Sphere(radius=1000, center=(gx_m, gy_m, gz_m))
        self.plotter.add_mesh(goal_marker, color="green")

        # Optionally draw the A* path
        if show_path and self.path_found is not None:
            self._add_astar_path_mesh(self.path_found)

        # Camera setup remains unchanged
        offset = 150000
        self.plotter.camera_position = [
            (x_m + offset, y_m + offset, z_m + offset),
            (x_m, y_m, z_m),
            (0, 0, 1)
        ]

        self.plotter.show_grid(
            xtitle="X (meters)",
            ytitle="Y (meters)",
            ztitle="Elevation (meters)",
        )

        if save_frames:
            return self.plotter.screenshot()
        else:
            self.plotter.show()

class NormalizeObservation(gym.ObservationWrapper):
    """
    Rescales LunarRover3DEnv observations to ~[-1, +1].
    We'll dynamically read (H, W) from self.env.dem_shape.
    """
    def __init__(self, env):
        super().__init__(env)

        # Original obs has shape=(15,)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )
        # Retrieve width, height from the underlying environment
        self.H, self.W = self.env.dem_shape  # (H, W)

    def observation(self, obs):
        """
        Obs layout is:
          [ x, y, x_dest, y_dest, current_incl,
            surrounding_inclinations(8),
            scaled_dist, scaled_angle ]
        => 15 total.
        """
        x, y, gx, gy = obs[0], obs[1], obs[2], obs[3]
        current_incl = obs[4]
        surrounding_incl = obs[5:13]  # 8 values
        scaled_dist   = obs[13]
        scaled_angle  = obs[14]

        # 1) Map x,y from [0, W-1], [0, H-1] => [-1,1]
        x_norm  = 2.0 * (x  / (self.W - 1)) - 1.0
        y_norm  = 2.0 * (y  / (self.H - 1)) - 1.0
        gx_norm = 2.0 * (gx / (self.W - 1)) - 1.0
        gy_norm = 2.0 * (gy / (self.H - 1)) - 1.0

        # 2) Inclinations from [-90..+90] => [-1..+1]
        current_incl_norm = current_incl / 90.0
        surrounding_incl_norm = surrounding_incl / 90.0  # vectorized

        # 3) scaled_dist in [0..1], map to [-1..+1]
        scaled_dist_norm = 2.0 * scaled_dist - 1.0

        # 4) scaled_angle is already in [-1..+1]
        scaled_angle_norm = scaled_angle

        # Reassemble
        obs_norm = np.array([
            x_norm, y_norm,
            gx_norm, gy_norm,
            current_incl_norm,
            *surrounding_incl_norm,
            scaled_dist_norm,
            scaled_angle_norm
        ], dtype=np.float32)

        return obs_norm

# -------------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    dem_file_path = "/Users/jbm/Desktop/Moon_Rover_SouthPole/src/map/LDEM_80S_20MPP_ADJ.tiff"
    subregion_window = (0, 6000, 0, 6000)
    env = LunarRover3DEnv(dem_file_path, subregion_window)

    # # Optionally run A* from spawn -> destination
    # spawn_rc = (int(round(env.spawn[1])), int(round(env.spawn[0])))
    # goal_rc = (int(round(env.destination[1])), int(round(env.destination[0])))
    # env.path_found = env.astar_path(spawn_rc, goal_rc)

    obs, _ = env.reset()
    env.render(show_path=True)