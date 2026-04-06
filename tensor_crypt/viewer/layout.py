import pygame


class LayoutManager:
    def __init__(self, viewer):
        self.viewer = viewer
        self.margin = 8

    def side_width(self):
        return max(340, min(450, int(self.viewer.Wpix * 0.25)))

    def hud_height(self):
        return 70

    def world_rect(self):
        m = self.margin
        return pygame.Rect(
            m,
            m,
            self.viewer.Wpix - self.side_width() - 2 * m,
            self.viewer.Hpix - self.hud_height() - 2 * m,
        )

    def side_rect(self):
        m, side_w = self.margin, self.side_width()
        return pygame.Rect(self.viewer.Wpix - side_w, m, side_w, self.viewer.Hpix - m * 2)

    def hud_rect(self):
        return pygame.Rect(
            self.margin,
            self.viewer.Hpix - self.hud_height() - self.margin,
            self.viewer.Wpix - self.side_width() - 2 * self.margin,
            self.hud_height(),
        )
