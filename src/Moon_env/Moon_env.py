import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv
from heapq import heappush, heappop
import math
from scipy.ndimage import gaussian_filter
import os

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
                 desired_distance_m=100000,
                 goal_radius_m=50):
        super().__init__()
        self.render_mode = render_mode
        self.max_slope_deg = max_slope_deg
        self.goal_radius_m = goal_radius_m          # Store as instance variable
        self.desired_distance_m = desired_distance_m

        # --- Load the DEM ---
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            # Pixel resolution (meters per pixel)
            self.x_res = abs(src.res[0])  # meters/pixel in x direction
            self.y_res = abs(src.res[1])  # meters/pixel in y direction

            if subregion_window is not None:
                row_start, row_end, col_start, col_end = subregion_window
                self.dem = dem_data[row_start:row_end, col_start:col_end]
            else:
                self.dem = dem_data

        # Optional smoothing to remove small terrain noise:
        if smooth_sigma is not None:
            self.dem = gaussian_filter(self.dem, sigma=smooth_sigma)

        self.dem_shape = self.dem.shape  # (rows, cols)
        self.dem_min = float(np.min(self.dem))
        self.dem_max = float(np.max(self.dem))

        # Interpolator for continuous (x, y)->z (note: row=y, col=x)
        y_coords = np.arange(self.dem_shape[0])
        x_coords = np.arange(self.dem_shape[1])
        self.dem_interp = RegularGridInterpolator((y_coords, x_coords), self.dem)

        # ---------------------------------------------------------------------
        # Observation Space:
        #   [ x, y, x_dest, y_dest, current_slope, surrounding_slopes(8) ]
        # => dimension = 2 + 2 + 1 + 8 = 13
        # x, y, x_dest, y_dest in [0, width-1], [0, height-1]
        # slope values in [0, 90]
        # ---------------------------------------------------------------------
        obs_low = np.array([0.0, 
                            0.0, 
                            0.0, 
                            0.0, 
                            0.0] + [0.0]*8, dtype=np.float32)
        obs_high = np.array([self.dem_shape[1]-1, 
                             self.dem_shape[0]-1, 
                             self.dem_shape[1]-1, 
                             self.dem_shape[0]-1, 
                             90.0] + [90.0]*8, dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action Space: [turn (radians), forward_distance (pixels)]
        #   turn in [-pi, pi], forward in [0, 1].
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0], dtype=np.float32),
            high=np.array([np.pi, 1.0], dtype=np.float32)
        )

        # Internal state
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
        center_y = (self.dem_shape[0] - 1) * 0.60
        self.spawn = (center_x, center_y)

        # Destination
        if destination is None:
            self.destination = self.compute_reachable_destination_from_spawn(self.spawn)
        else:
            self.destination = destination
            # Compute path for predefined destination
            start_rc = (int(round(self.spawn[1])), int(round(self.spawn[0])))
            goal_rc = (int(round(self.destination[1])), int(round(self.destination[0])))
            self.path_found = self.astar_path(start_rc, goal_rc)

        print(f"Spawn: {self.spawn}, Destination: {self.destination}")

        # For optional path rendering
        # self.path_found = None

        # PyVista plotter
        self.plotter = None
        
        print(f"DEM resolution: x={self.x_res} m/px, y={self.y_res} m/px")
        print(f"Subregion size (meters): {self.dem_shape[0] * self.y_res / 1000} km x {self.dem_shape[1] * self.x_res / 1000} km")

    # -------------------------------------------------------------------------
    # Slope / Height Utilities
    # -------------------------------------------------------------------------
    def get_height(self, px, py):
        """
        Return the height at pixel coords (px, py).
        Because the interpolator expects (row, col) = (y, x).
        """
        point = np.array([[py, px]], dtype=np.float32)
        return float(self.dem_interp(point)[0])

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

    def get_slope(self, x1, y1, x2, y2):
        """
        Compute the slope in degrees from (x1, y1) to (x2, y2) in pixel coords,
        using physical distances in meters. Returns signed slope (negative = downhill).
        """
        # Convert pixel coords to meters - consistent conversion
        x1_m = x1 * self.x_res
        y1_m = y1 * self.y_res
        x2_m = x2 * self.x_res
        y2_m = y2 * self.y_res

        # Horizontal distance
        dx_m = x2_m - x1_m
        dy_m = y2_m - y1_m
        horizontal_dist_m = math.sqrt(dx_m**2 + dy_m**2)

        if horizontal_dist_m < 1e-9:
            return 0.0  # Avoid division by zero

        # Get heights at these positions
        z1 = self.get_height(x1, y1)
        z2 = self.get_height(x2, y2)
        dz = z2 - z1  # Retain sign for downhill/uphill

        slope_rad = math.atan(dz / horizontal_dist_m)
        slope_deg = math.degrees(slope_rad)
        return slope_deg  # Signed value
    
    def get_lateral_inclination(self, x, y, theta, side_px=1.0):
        """
        Returns the slope in the direction perpendicular to the rover’s heading.
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
        """
        Builds the observation vector:
           [ x, y, x_dest, y_dest, current_incl, surrounding_incl(8) ]
        """
        x, y, theta = self.state
        gx, gy = self.goal

        # Slopes
        current_incl = self.get_current_inclination(x, y, theta, forward_px=1.0)
        surrounding_incl_list = self.get_surrounding_inclinations(x, y, dist_px=1.0)
        # Flatten the 8 surrounding slopes
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

    def get_transition_lateral_slope(self, x1, y1, x2, y2, side_px=1.0):
        """
        In a grid-based A*, we have no explicit 'theta'. We treat the direction
        from (x1,y1) to (x2,y2) as the heading, then check the slope 90° from
        that heading at the new cell (x2,y2).
        """
        dx = x2 - x1
        dy = y2 - y1
        heading_theta = math.atan2(dy, dx)  # direction of travel
        lateral_theta = heading_theta + np.pi / 2.0

        # We'll measure the slope from the new cell outward to the side.
        side_x = x2 + side_px * math.cos(lateral_theta)
        side_y = y2 + side_px * math.sin(lateral_theta)

        # clamp
        side_x = np.clip(side_x, 0, self.dem_shape[1] - 1)
        side_y = np.clip(side_y, 0, self.dem_shape[0] - 1)

        return self.get_slope(x2, y2, side_x, side_y)

    def astar_path(self, start, goal):
        """
        A* over the 2D grid (row, col) = (y, x), with 8 neighbors.
        We now also check the lateral slope implied by the step from current->neighbor.
        """
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def distance(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_set = []
        heappush(open_set, (distance(start, goal), 0.0, start))
        came_from = {}
        cost_so_far = {start: 0.0}

        while open_set:
            _, current_cost, current = heappop(open_set)
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            cy, cx = current
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                # boundary check
                if ny < 0 or ny >= self.dem_shape[0] or nx < 0 or nx >= self.dem_shape[1]:
                    continue

                # Forward slope: from current cell (cx,cy) to neighbor (nx,ny)
                slope_deg = self.get_slope(cx, cy, nx, ny)

                # Lateral slope: interpret orientation from (cx,cy)->(nx,ny)
                side_slope_deg = self.get_transition_lateral_slope(cx, cy, nx, ny)

                # If either slope is too steep, skip
                if (abs(slope_deg) > self.max_slope_deg) or (abs(side_slope_deg) > self.max_slope_deg):
                    continue

                new_cost = current_cost + distance(current, (ny, nx))
                if (ny, nx) not in cost_so_far or new_cost < cost_so_far[(ny, nx)]:
                    cost_so_far[(ny, nx)] = new_cost
                    priority = new_cost + distance((ny, nx), goal)
                    heappush(open_set, (priority, new_cost, (ny, nx)))
                    came_from[(ny, nx)] = current

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
        """
        Attempt to find a destination that is reachable by A* within slope limit.
        The computed A* path is stored in self.path_found for later visualization.
        """
        desired_distance_px_x = self.desired_distance_m / self.x_res  # Use hyperparameter
        desired_distance_px_y = self.desired_distance_m / self.y_res  # Use hyperparameter
        angles = np.linspace(0, 360, 36)

        for angle_deg in angles:
            rad = math.radians(angle_deg)
            cx = spawn[0] + desired_distance_px_x * math.cos(rad)
            cy = spawn[1] + desired_distance_px_y * math.sin(rad)
            # Clamp to DEM boundaries
            cx = np.clip(cx, 0, self.dem_shape[1] - 1)
            cy = np.clip(cy, 0, self.dem_shape[0] - 1)
            candidate = (cx, cy)

            path = self.check_path_possible(spawn, candidate)
            if path is not None:
                print("Selected reachable destination:", candidate)
                # Store the computed path for visualization
                self.path_found = path
                return candidate

        # If none found, return the spawn as fallback
        return spawn

    # -------------------------------------------------------------------------
    # Gym Interface
    # -------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # State = [x, y, theta]
        start_x, start_y = self.spawn
        self.state = np.array([start_x, start_y, 0.0], dtype=np.float32)

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

        # Update heading
        theta += turn

        # Move forward (in pixel coords)
        new_x = x + forward * math.cos(theta)
        new_y = y + forward * math.sin(theta)
        # clamp
        new_x = np.clip(new_x, 0, self.dem_shape[1] - 1)
        new_y = np.clip(new_y, 0, self.dem_shape[0] - 1)

        self.state = np.array([new_x, new_y, theta], dtype=np.float32)
        self.current_step += 1

        # Observation
        obs = self.get_observation()

        # 1) Forward slope
        current_incl = obs[4]

        # 2) Lateral slope
        lateral_incl = self.get_lateral_inclination(new_x, new_y, theta, side_px=1.0)

        # Crash check if forward OR lateral slope exceed ±max_slope_deg
        if abs(current_incl) > self.max_slope_deg or abs(lateral_incl) > self.max_slope_deg:
            reward = -50.0
            done = True
            return obs, reward, done, False, {}

        # Basic shaping reward
        reward = -0.1

        # Goal check
        gx, gy = self.goal
        dx_m = (new_x - gx) * self.x_res
        dy_m = (new_y - gy) * self.y_res
        dist_to_goal_m = np.linalg.norm([dx_m, dy_m])
        if dist_to_goal_m < self.goal_radius_m:
            reward += 100.0
            done = True
        else:
            done = (self.current_step >= self.max_steps)

        return obs, reward, done, False, {}

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
    subregion_window = (0, 10000, 0, 10000)
    env = LunarRover3DEnv(dem_file_path, subregion_window)

    # # Optionally run A* from spawn -> destination
    # spawn_rc = (int(round(env.spawn[1])), int(round(env.spawn[0])))
    # goal_rc = (int(round(env.destination[1])), int(round(env.destination[0])))
    # env.path_found = env.astar_path(spawn_rc, goal_rc)

    obs, _ = env.reset()
    env.render(show_path=True)