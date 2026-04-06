import pygame


class TextCache:
    def __init__(self, font_name="consolas", sizes=None):
        if sizes is None:
            sizes = [12, 13, 16, 18, 24]
        self.fonts = {}
        for sz in sizes:
            try:
                self.fonts[sz] = pygame.font.SysFont(font_name, sz)
            except Exception:
                self.fonts[sz] = pygame.font.Font(None, sz + 2)
        self.cache = {}

    def render(self, text, size, color, aa=True):
        key = (text, size, color, aa)
        if key not in self.cache:
            self.cache[key] = self.fonts[size].render(text, aa, color)
        return self.cache[key]
