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
                goal_radius_m=50):
        super().__init__()
        self.render_mode = render_mode
        self.max_slope_deg = max_slope_deg
        self.goal_radius_m = goal_radius_m
        self.desired_distance_m = desired_distance_m
        self.path_found = None  # Make sure this is defined

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

        # Interpolator for continuous (row=y, col=x)
        y_coords = np.arange(self.dem_shape[0])
        x_coords = np.arange(self.dem_shape[1])
        self.dem_interp = RegularGridInterpolator((y_coords, x_coords), self.dem)

        # ---------- NEW CODE for gradients in “m/m” instead of “m/pixel” ----------
        # 1) get the raw gradient in pixel units
        raw_grad_x, raw_grad_y = np.gradient(self.dem)
        # 2) convert them to "vertical meters per horizontal meter"
        grad_x = raw_grad_x / self.y_res   # Because raw_grad_x was dZ/dRow, row→vertical pixel axis
        grad_y = raw_grad_y / self.x_res   # Because raw_grad_y was dZ/dCol, col→horizontal pixel axis
        self.grad_x = grad_x
        self.grad_y = grad_y
        # --------------------------------------------------------------------------

        # Define the observation and action spaces
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0] + [0.0]*8, dtype=np.float32)
        obs_high = np.array(
            [self.dem_shape[1]-1, 
            self.dem_shape[0]-1,
            self.dem_shape[1]-1, 
            self.dem_shape[0]-1,
            90.0] + [90.0]*8, dtype=np.float32
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0], dtype=np.float32),
            high=np.array([np.pi, 1.0], dtype=np.float32)
        )

        self.state = None  # [x, y, theta]
        self.max_steps = 1000
        self.current_step = 0

        # Precompute grid in meters for rendering
        X, Y = np.meshgrid(
            np.arange(self.dem_shape[1]) * self.x_res,
            np.arange(self.dem_shape[0]) * self.y_res
        )
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

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

    # -------------------------------------------------------------------------
    # Slope / Height Utilities
    # -------------------------------------------------------------------------
    @lru_cache(maxsize=10000)
    def get_height(self, px, py):
        """Faster bilinear interpolation with caching"""
        # Convert to integer coordinates for caching
        px, py = float(px), float(py)
        
        # Integer and fractional parts
        x0, y0 = int(px), int(py)
        x1, y1 = min(x0 + 1, self.dem_shape[1] - 1), min(y0 + 1, self.dem_shape[0] - 1)
        dx, dy = px - x0, py - y0
        
        # Bilinear interpolation
        return ((1-dx)*(1-dy)*self.dem[y0, x0] + 
                dx*(1-dy)*self.dem[y0, x1] + 
                (1-dx)*dy*self.dem[y1, x0] + 
                dx*dy*self.dem[y1, x1])

    # def get_slope(self, x1, y1, x2, y2):
    #     """
    #     Compute the absolute slope in degrees from (x1, y1) to (x2, y2) in pixel coords,
    #     using physical distances in meters.
    #     """
    #     # Convert pixel coords to meters
    #     x1_m = x1 * self.x_res
    #     y1_m = y1 * self.y_res
    #     x2_m = x2 * self.x_res
    #     y2_m = y2 * self.y_res

    #     # Horizontal distance
    #     dx_m = x2_m - x1_m
    #     dy_m = y2_m - y1_m
    #     horizontal_dist_m = math.sqrt(dx_m**2 + dy_m**2)

    #     if horizontal_dist_m < 1e-9:
    #         return 0.0  # no movement => slope is 0

    #     z1 = self.get_height(x1, y1)
    #     z2 = self.get_height(x2, y2)
    #     dz = z2 - z1

    #     slope_rad = math.atan(abs(dz) / horizontal_dist_m)
    #     slope_deg = math.degrees(slope_rad)
    #     return slope_deg

    @lru_cache(maxsize=10000)
    def get_slope(self, x1, y1, x2, y2):
        """Compute slope using precomputed gradients with caching"""
        # Convert to float for consistent cache keys
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        
        # Use average gradient at the points
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        
        # Get integer coordinates for sampling gradients
        xi, yi = int(x_mid), int(y_mid)
        xi = np.clip(xi, 0, self.dem_shape[1] - 2)
        yi = np.clip(yi, 0, self.dem_shape[0] - 2)
        
        # Get direction vector in meters
        dx_m = (x2 - x1) * self.x_res
        dy_m = (y2 - y1) * self.y_res
        dist_m = math.sqrt(dx_m**2 + dy_m**2)
        
        if dist_m < 1e-9:
            return 0.0
        
        # Normalize direction vector
        dx_norm = dx_m / dist_m
        dy_norm = dy_m / dist_m
        
        # Dot product with gradient gives slope in that direction
        slope = dx_norm * self.grad_x[yi, xi] + dy_norm * self.grad_y[yi, xi]
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
        if self.check_all_directions_slopes(initial_spawn_x, initial_spawn_y):
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
    
    def get_lateral_inclination(self, x, y, theta, side_px=1.0):
        """
        Returns the slope in the direction perpendicular to the rover's heading.
        side_px indicates how far to look 'sideways' from the current position.
        """
        # Perpendicular heading: theta + 90°
        lateral_theta = theta + np.pi / 2.0
        
        # Destination pixel to the side
        x2 = x + side_px * math.cos(lateral_theta)
        y2 = y + side_px * math.sin(lateral_theta)
        
        # Clamp to DEM boundaries
        x2 = np.clip(x2, 0, self.dem_shape[1] - 1)
        y2 = np.clip(y2, 0, self.dem_shape[0] - 1)
        
        # Reuse the same slope calc
        return self.get_slope(x, y, x2, y2)


    def get_current_inclination(self, x, y, theta, forward_px=1.0):
        """
        Slope in degrees in the rover's heading direction.
        We'll check a small distance 'forward_px' in front.
        """
        x2 = x + forward_px * math.cos(theta)
        y2 = y + forward_px * math.sin(theta)
        # clamp to DEM boundaries
        x2 = np.clip(x2, 0, self.dem_shape[1] - 1)
        y2 = np.clip(y2, 0, self.dem_shape[0] - 1)
        return self.get_slope(x, y, x2, y2)

    # def get_surrounding_inclinations(self, x, y, dist_px=1.0):
    #     """
    #     Return the slopes in 8 surrounding directions (at 45° increments).
    #     e.g. 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°.
    #     """
    #     directions_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    #     slopes = []
    #     for d in directions_deg:
    #         theta = math.radians(d)
    #         x2 = x + dist_px * math.cos(theta)
    #         y2 = y + dist_px * math.sin(theta)
    #         # clamp
    #         x2 = np.clip(x2, 0, self.dem_shape[1] - 1)
    #         y2 = np.clip(y2, 0, self.dem_shape[0] - 1)
    #         slope = self.get_slope(x, y, x2, y2)
    #         slopes.append(slope)
    #     return slopes

    def get_surrounding_inclinations(self, x, y, dist_px=1.0):
        """
        Return slopes in 8 directions (signed degrees).
        Example: -15° = downhill, +15° = uphill.
        """
        directions_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        slopes = []
        for d in directions_deg:
            theta = math.radians(d)
            x2 = x + dist_px * math.cos(theta)
            y2 = y + dist_px * math.sin(theta)
            x2 = np.clip(x2, 0, self.dem_shape[1] - 1)
            y2 = np.clip(y2, 0, self.dem_shape[0] - 1)
            slope = self.get_slope(x, y, x2, y2)  # Signed slope
            slopes.append(slope)
        return slopes

    # -------------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------------
    def get_observation(self):
        """More efficient observation building"""
        x, y, theta = self.state
        gx, gy = self.goal
        
        # Reuse the same forward vector for current_incl
        forward_px = 1.0
        dx = forward_px * math.cos(theta)
        dy = forward_px * math.sin(theta)
        x2 = np.clip(x + dx, 0, self.dem_shape[1] - 1)
        y2 = np.clip(y + dy, 0, self.dem_shape[0] - 1)
        
        # Get current inclination
        current_incl = self.get_slope(x, y, x2, y2)
        
        # Get surrounding inclinations
        surrounding_incl_list = self.get_surrounding_inclinations(x, y, dist_px=1.0)
        
        # Combine into observation vector
        obs = np.array([
            x,
            y,
            gx,
            gy,
            current_incl
        ] + surrounding_incl_list, dtype=np.float32)
        
        return obs

    # -------------------------------------------------------------------------
    # A* with slope constraint
    # -------------------------------------------------------------------------
    # def astar_path(self, start, goal):
    #     """
    #     A* over the 2D grid (row, col) = (y, x), with 8 neighbors.
    #     The slope constraint is that it must be <= max_slope_deg in absolute value.
    #     Measure slope in meters, as done in get_slope().
    #     """
    #     neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
    #                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    #     def distance(a, b):
    #         return np.linalg.norm(np.array(a) - np.array(b))

    #     # Store them in open_set as (priority, cost, cell)
    #     open_set = []
    #     heappush(open_set, (distance(start, goal), 0.0, start))
    #     came_from = {}
    #     cost_so_far = {start: 0.0}

    #     # Use the tangent threshold for slope check
    #     tan_thresh = math.tan(math.radians(self.max_slope_deg))

    #     while open_set:
    #         _, current_cost, current = heappop(open_set)
    #         if current == goal:
    #             # Reconstruct path
    #             path = []
    #             while current in came_from:
    #                 path.append(current)
    #                 current = came_from[current]
    #             path.append(start)
    #             path.reverse()
    #             return path

    #         cy, cx = current
    #         for dy, dx in neighbors:
    #             ny, nx = cy + dy, cx + dx
    #             # boundary check
    #             if ny < 0 or ny >= self.dem_shape[0] or nx < 0 or nx >= self.dem_shape[1]:
    #                 continue

    #             # Check slope
    #             # Convert (cy,cx)->(x,y) in pixel coords => (cx, cy)
    #             slope_deg = self.get_slope(cx, cy, nx, ny)
    #             if slope_deg > self.max_slope_deg:
    #                 continue

    #             new_cost = current_cost + distance(current, (ny, nx))
    #             if (ny, nx) not in cost_so_far or new_cost < cost_so_far[(ny, nx)]:
    #                 cost_so_far[(ny, nx)] = new_cost
    #                 priority = new_cost + distance((ny, nx), goal)
    #                 heappush(open_set, (priority, new_cost, (ny, nx)))
    #                 came_from[(ny, nx)] = current

    #     return None  # No path found

    def check_all_directions_slopes(self, x, y, dist_px=1.0):
        """Vectorized and optimized slope check in all directions"""
        # Cache common values used in the function
        x_res = self.x_res
        max_slope_deg = self.max_slope_deg
        
        # Get center height once
        center_height = self.get_height(x, y)
        
        # Only check 4 cardinal directions first (faster)
        cardinal_dirs = [(0, dist_px), (dist_px, 0), (0, -dist_px), (-dist_px, 0)]
        for dx, dy in cardinal_dirs:
            x2 = np.clip(x + dx, 0, self.dem_shape[1] - 1)
            y2 = np.clip(y + dy, 0, self.dem_shape[0] - 1)
            
            # Get height
            height = self.get_height(x2, y2)
            
            # Calculate slope in meters
            horizontal_dist_m = dist_px * x_res  # Assuming square pixels
            slope_deg = math.degrees(math.atan(abs(height - center_height) / horizontal_dist_m))
            
            if slope_deg > max_slope_deg:
                return False
        
        # Only check diagonals if cardinal directions pass
        diagonal_dirs = [(dist_px, dist_px), (dist_px, -dist_px), 
                        (-dist_px, dist_px), (-dist_px, -dist_px)]
        
        for dx, dy in diagonal_dirs:
            x2 = np.clip(x + dx, 0, self.dem_shape[1] - 1)
            y2 = np.clip(y + dy, 0, self.dem_shape[0] - 1)
            
            # Get height
            height = self.get_height(x2, y2)
            
            # Calculate slope in meters (diagonal distance)
            horizontal_dist_m = dist_px * x_res * math.sqrt(2)  # Diagonal distance
            slope_deg = math.degrees(math.atan(abs(height - center_height) / horizontal_dist_m))
            
            if slope_deg > max_slope_deg:
                return False
        
        return True

    
    def is_transition_safe(self, x1, y1, x2, y2, step_px=0.5):
        """
        Optimized version that checks if the path from (x1,y1) to (x2,y2) is safe
        """
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        
        # Quick check: if distance is very small, just check endpoints
        if distance < step_px:
            return self.check_all_directions_slopes(x1, y1) and self.check_all_directions_slopes(x2, y2)
        
        steps = max(1, int(distance / step_px))
        
        # Check start and end first - most common failure points
        if not (self.check_all_directions_slopes(x1, y1) and self.check_all_directions_slopes(x2, y2)):
            return False
        
        # If only a few steps, check them all
        if steps <= 3:
            for i in range(1, steps):
                t = i / steps
                xi = x1 + t * dx
                yi = y1 + t * dy
                if not self.check_all_directions_slopes(xi, yi):
                    return False
            return True
        
        # For longer paths, check fewer intermediate points
        # Check middle point and 1/4, 3/4 points
        checkpoints = [0.25, 0.5, 0.75]
        for t in checkpoints:
            xi = x1 + t * dx
            yi = y1 + t * dy
            if not self.check_all_directions_slopes(xi, yi):
                return False
        
        return True

    def astar_path(self, start, goal):
        """Highly optimized A* implementation"""
        # Early exit if start == goal
        if start == goal:
            return [start]
        
        # Use a more efficient distance calculation
        def octile_distance(a, b):
            dx = abs(a[1] - b[1])
            dy = abs(a[0] - b[0])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        
        # Create numpy array for visited nodes (much faster than dict lookups)
        visited = np.zeros(self.dem_shape, dtype=bool)
        # Create cost array initialized to infinity
        cost_array = np.full(self.dem_shape, np.inf)
        cost_array[start] = 0
        
        # Cardinal and diagonal movements with precalculated costs
        neighbors = np.array([(-1, 0), (1, 0), (0, -1), (0, 1),
                            (-1, -1), (-1, 1), (1, -1), (1, 1)])
        costs = np.array([1.0, 1.0, 1.0, 1.0, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)])
        
        # Initialize priority queue with (f_score, tiebreaker, node_y, node_x)
        open_set = [(octile_distance(start, goal), 0, start[0], start[1])]
        # Dictionary for backtracking
        came_from = {}
        counter = 0  # Tiebreaker for equal f-scores
        
        # Precompute goal coordinates for faster access
        goal_y, goal_x = goal
        
        # For slope safety check optimization
        max_slope_tan = math.tan(math.radians(self.max_slope_deg))
        
        while open_set:
            # Pop node with lowest f_score
            _, _, cy, cx = heappop(open_set)
            current = (cy, cx)
            
            # Goal check
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Faster than path.reverse()
            
            # Mark as visited
            visited[cy, cx] = True
            current_cost = cost_array[cy, cx]
            
            # Check all neighbors at once
            for i, (dy, dx) in enumerate(neighbors):
                ny, nx = cy + dy, cx + dx
                
                # Skip if out of bounds or already visited
                if (ny < 0 or ny >= self.dem_shape[0] or 
                    nx < 0 or nx >= self.dem_shape[1] or 
                    visited[ny, nx]):
                    continue
                
                # Quick slope check using precomputed gradients
                # This avoids calling the expensive is_transition_safe for all neighbors
                x_mid = (cx + nx) / 2
                y_mid = (cy + ny) / 2
                xi, yi = int(x_mid), int(y_mid)
                xi = min(max(xi, 0), self.dem_shape[1] - 2)
                yi = min(max(yi, 0), self.dem_shape[0] - 2)
                
                # Direction vector in pixels
                dx_px = nx - cx
                dy_px = ny - cy
                dist_px = math.sqrt(dx_px**2 + dy_px**2)
                
                # Convert to meters
                dx_m = dx_px * self.x_res
                dy_m = dy_px * self.y_res
                dist_m = math.sqrt(dx_m**2 + dy_m**2)
                
                # Normalize
                dx_norm = dx_m / dist_m if dist_m > 1e-9 else 0
                dy_norm = dy_m / dist_m if dist_m > 1e-9 else 0
                
                # Quick slope check
                slope = dx_norm * self.grad_x[yi, xi] + dy_norm * self.grad_y[yi, xi]
                if abs(slope) > max_slope_tan:
                    continue
                
                # Only perform expensive check if quick check passes
                if not self.is_transition_safe(cx, cy, nx, ny):
                    continue
                
                # Calculate new cost
                new_cost = current_cost + costs[i]
                
                # If better path found
                if new_cost < cost_array[ny, nx]:
                    # Update tracking information
                    came_from[(ny, nx)] = current
                    cost_array[ny, nx] = new_cost
                    f_score = new_cost + octile_distance((ny, nx), goal)
                    counter += 1
                    heappush(open_set, (f_score, counter, ny, nx))
        
        return None  # No path found

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

    # def step(self, action):
    #     x, y, theta = self.state
    #     turn, forward = action

    #     # Update heading
    #     theta += turn

    #     # Move forward (in pixel coords)
    #     new_x = x + forward * math.cos(theta)
    #     new_y = y + forward * math.sin(theta)
    #     # clamp
    #     new_x = np.clip(new_x, 0, self.dem_shape[1] - 1)
    #     new_y = np.clip(new_y, 0, self.dem_shape[0] - 1)

    #     self.state = np.array([new_x, new_y, theta], dtype=np.float32)
    #     self.current_step += 1

    #     # Observation
    #     obs = self.get_observation()

    #     # Crash check if slope > 25° in the heading direction
    #     current_incl = obs[4]  # the single "current_incl" in the vector
    #     if current_incl > self.max_slope_deg:
    #         reward = -50.0
    #         done = True
    #         return obs, reward, done, False, {}

    #     # Basic shaping reward
    #     reward = -0.1

    #     # Goal check
    #     gx, gy = self.goal
    #     dist_to_goal = np.linalg.norm([new_x - gx, new_y - gy])
    #     if dist_to_goal < 5.0:
    #         reward += 100.0
    #         done = True
    #     else:
    #         done = (self.current_step >= self.max_steps)

    #     return obs, reward, done, False, {}

    def step(self, action):
        x, y, theta = self.state
        turn, forward = action
        theta += turn

        # Skip interpolation for small movements
        if forward <= 0.1:
            new_x = x + forward * math.cos(theta)
            new_y = y + forward * math.sin(theta)
            new_x = np.clip(new_x, 0, self.dem_shape[1] - 1)
            new_y = np.clip(new_y, 0, self.dem_shape[0] - 1)
            
            # Check slopes at the destination point
            if not self.check_all_directions_slopes(new_x, new_y):
                self.state = np.array([new_x, new_y, theta], dtype=np.float32)
                return self.get_observation(), -50.0, True, False, {}
                
            self.state = np.array([new_x, new_y, theta], dtype=np.float32)
        else:
            # Interpolate movement into smaller steps for longer movements
            step_size = 0.2  # Larger step size than before (0.1)
            steps = max(int(forward / step_size), 1)
            
            for i in range(steps):
                partial_forward = (i + 1) * step_size
                new_x = x + min(partial_forward, forward) * math.cos(theta)
                new_y = y + min(partial_forward, forward) * math.sin(theta)
                new_x = np.clip(new_x, 0, self.dem_shape[1] - 1)
                new_y = np.clip(new_y, 0, self.dem_shape[0] - 1)
                
                # Check slopes
                if not self.check_all_directions_slopes(new_x, new_y):
                    self.state = np.array([new_x, new_y, theta], dtype=np.float32)
                    return self.get_observation(), -50.0, True, False, {}
                    
                # Update position for next step
                x, y = new_x, new_y
                
            # Update final state
            self.state = np.array([x, y, theta], dtype=np.float32)

        # Reward calculation
        reward = -0.1
        
        # Check goal
        gx, gy = self.goal
        dx_m = (x - gx) * self.x_res
        dy_m = (y - gy) * self.y_res
        dist_to_goal_m = np.linalg.norm([dx_m, dy_m])
        
        if dist_to_goal_m < self.goal_radius_m:
            reward += 100.0
            done = True
        else:
            self.current_step += 1
            done = (self.current_step >= self.max_steps)

        return self.get_observation(), reward, done, False, {}

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    def render(self, mode='human', show_path=False, save_frames=False):
        """
        Renders the environment using PyVista. 
        If 'show_path=True' and self.path_found is not None, draws the path in 3D.
        """
        if self.plotter is None:
            self.plotter = pv.Plotter(window_size=(1024, 768))
        else:
            self.plotter.clear()

        # Create structured grid
        grid = pv.StructuredGrid(self.X, self.Y, self.dem)
        self.plotter.add_mesh(grid, cmap="Blues_r", show_scalar_bar=True)

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

        # Camera
        offset = 150000
        self.plotter.camera_position = [
            (x_m + offset, y_m + offset, z_m + offset),  # camera location
            (x_m, y_m, z_m),                             # look at rover
            (0, 0, 1)                                    # up vector
        ]

        # self.plotter = pv.Plotter(window_size=(1024, 768))
        #self.plotter.enable_terrain_style()  # Optional: Better visualization
        self.plotter.show_grid(
            xtitle="X (meters)",
            ytitle="Y (meters)",
            ztitle="Elevation (meters)",
        )

        if save_frames:
            return self.plotter.screenshot()
        else:
            self.plotter.show()

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