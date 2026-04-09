class Camera:
    def __init__(self, screen_width, screen_height, grid_w, grid_h):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.min_zoom = 0.5
        self.max_zoom = 64.0
        self.fit_to_world()

    def update_screen_size(self, w, h):
        self.screen_width = w
        self.screen_height = h

    def fit_to_world(self):
        """Set zoom and offsets so the full world fits and stays centered."""
        if self.screen_width <= 0 or self.screen_height <= 0:
            self.cell_px = 1.0
            self.offset_x = 0.0
            self.offset_y = 0.0
            return

        scale_x = self.screen_width / self.grid_w
        scale_y = self.screen_height / self.grid_h
        self.cell_px = max(self.min_zoom, min(self.max_zoom, min(scale_x, scale_y)))

        visible_w = self.screen_width / self.cell_px
        visible_h = self.screen_height / self.cell_px
        self.offset_x = (self.grid_w - visible_w) / 2.0
        self.offset_y = (self.grid_h - visible_h) / 2.0
        self._clamp_offsets()

    def _clamp_offsets(self):
        """Keep the world in view while allowing bounded panning."""
        visible_w = self.screen_width / max(self.cell_px, 0.01)
        visible_h = self.screen_height / max(self.cell_px, 0.01)

        min_ox = -visible_w * 0.5
        max_ox = self.grid_w - visible_w * 0.5
        min_oy = -visible_h * 0.5
        max_oy = self.grid_h - visible_h * 0.5

        self.offset_x = max(min_ox, min(max_ox, self.offset_x))
        self.offset_y = max(min_oy, min(max_oy, self.offset_y))

    def pan(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
        self._clamp_offsets()

    def zoom_at(self, factor, mx, my):
        old_cell_px = self.cell_px
        world_x = self.offset_x + mx / old_cell_px
        world_y = self.offset_y + my / old_cell_px
        self.cell_px = max(self.min_zoom, min(self.max_zoom, self.cell_px * factor))
        new_world_x = self.offset_x + mx / self.cell_px
        new_world_y = self.offset_y + my / self.cell_px
        self.offset_x += world_x - new_world_x
        self.offset_y += world_y - new_world_y
        self._clamp_offsets()

    def world_to_screen(self, x, y):
        cx = (x - self.offset_x) * self.cell_px
        cy = (y - self.offset_y) * self.cell_px
        return int(cx), int(cy)

    def screen_to_world(self, cx, cy):
        gx = int(self.offset_x + cx / self.cell_px)
        gy = int(self.offset_y + cy / self.cell_px)
        gx = max(0, min(self.grid_w - 1, gx))
        gy = max(0, min(self.grid_h - 1, gy))
        return gx, gy

    def screen_to_world_float(self, cx, cy):
        wx = self.offset_x + cx / max(self.cell_px, 0.01)
        wy = self.offset_y + cy / max(self.cell_px, 0.01)
        return wx, wy
