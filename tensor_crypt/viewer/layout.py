import pygame


class LayoutManager:
    def __init__(self, viewer):
        self.viewer = viewer
        self.margin = 8

    def side_width(self):
        return max(340, min(450, int(self.viewer.Wpix * 0.25)))

    def hud_height(self):
        return 96

    def world_rect(self):
        m = self.margin
        sw = self.side_width()
        hh = self.hud_height()
        available_w = self.viewer.Wpix - sw - 2 * m
        available_h = self.viewer.Hpix - hh - 2 * m
        return pygame.Rect(m, m, max(1, available_w), max(1, available_h))

    def side_rect(self):
        m = self.margin
        sw = self.side_width()
        return pygame.Rect(self.viewer.Wpix - sw, m, sw, self.viewer.Hpix - m * 2)

    def hud_rect(self):
        m = self.margin
        sw = self.side_width()
        hh = self.hud_height()
        return pygame.Rect(m, self.viewer.Hpix - hh - m, self.viewer.Wpix - sw - 2 * m, hh)
