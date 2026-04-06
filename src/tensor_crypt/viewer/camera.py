class Camera:
    def __init__(self, screen_width, screen_height, grid_w, grid_h):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_px = min(screen_width / grid_w, screen_height / grid_h)
        self.offset_x = (grid_w - screen_width / self.cell_px) / 2
        self.offset_y = (grid_h - screen_height / self.cell_px) / 2
        self.min_zoom = 0.5
        self.max_zoom = 64

    def update_screen_size(self, w, h):
        self.screen_width = w
        self.screen_height = h

    def pan(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy

    def zoom_at(self, factor, mx, my):
        old_cell_px = self.cell_px
        world_x = self.offset_x + mx / old_cell_px
        world_y = self.offset_y + my / old_cell_px
        self.cell_px = max(self.min_zoom, min(self.max_zoom, self.cell_px * factor))
        new_world_x = self.offset_x + mx / self.cell_px
        new_world_y = self.offset_y + my / self.cell_px
        self.offset_x += world_x - new_world_x
        self.offset_y += world_y - new_world_y

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
