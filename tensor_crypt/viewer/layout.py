
import pygame


class LayoutManager:
    """Authoritative viewer geometry for the world, HUD, and side panel."""

    WORLD_MIN_WIDTH = 360
    SIDE_WIDTH_RATIO = 0.28
    SIDE_WIDTH_MIN = 280
    SIDE_WIDTH_MAX = 420
    SIDE_WIDTH_HARD_MIN = 220
    HUD_HEIGHT_REGULAR = 96
    HUD_HEIGHT_DENSE = 84
    HUD_HEIGHT_NARROW = 116
    PANEL_PADDING_REGULAR = 12
    PANEL_PADDING_DENSE = 10

    def __init__(self, viewer):
        self.viewer = viewer
        self.margin = 8

    def is_dense(self) -> bool:
        return self.viewer.Wpix < 1180 or self.viewer.Hpix < 760

    def panel_padding(self) -> int:
        return self.PANEL_PADDING_DENSE if self.is_dense() else self.PANEL_PADDING_REGULAR

    def side_width(self) -> int:
        desired = int(self.viewer.Wpix * self.SIDE_WIDTH_RATIO)
        desired = max(self.SIDE_WIDTH_MIN, min(self.SIDE_WIDTH_MAX, desired))
        max_side_width = self.viewer.Wpix - (self.margin * 3) - self.WORLD_MIN_WIDTH
        desired = min(desired, max_side_width)
        return max(self.SIDE_WIDTH_HARD_MIN, desired)

    def hud_height(self) -> int:
        if self.viewer.Wpix < 960:
            return self.HUD_HEIGHT_NARROW
        if self.is_dense():
            return self.HUD_HEIGHT_DENSE
        return self.HUD_HEIGHT_REGULAR

    def world_rect(self) -> pygame.Rect:
        sw = self.side_width()
        hh = self.hud_height()
        width = max(1, self.viewer.Wpix - sw - (self.margin * 3))
        height = max(1, self.viewer.Hpix - hh - (self.margin * 2))
        return pygame.Rect(self.margin, self.margin, width, height)

    def side_rect(self) -> pygame.Rect:
        world = self.world_rect()
        x = world.right + self.margin
        width = max(1, self.viewer.Wpix - x - self.margin)
        return pygame.Rect(x, self.margin, width, max(1, self.viewer.Hpix - (self.margin * 2)))

    def hud_rect(self) -> pygame.Rect:
        world = self.world_rect()
        hh = self.hud_height()
        return pygame.Rect(world.x, self.viewer.Hpix - hh - self.margin, world.width, hh)

    def content_rect(self, rect: pygame.Rect, padding: int | None = None) -> pygame.Rect:
        pad = self.panel_padding() if padding is None else max(0, int(padding))
        width = max(1, rect.width - (pad * 2))
        height = max(1, rect.height - (pad * 2))
        return pygame.Rect(rect.x + pad, rect.y + pad, width, height)