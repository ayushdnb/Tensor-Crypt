
import pygame


class TextCache:
    def __init__(self, font_name="consolas", sizes=None):
        self.font_name = font_name
        if sizes is None:
            sizes = [11, 12, 13, 14, 16, 18, 24]
        self.fonts = {}
        for sz in sizes:
            self.fonts[sz] = self._build_font(sz)
        self.cache = {}

    def _build_font(self, size):
        try:
            return pygame.font.SysFont(self.font_name, size)
        except Exception:
            return pygame.font.Font(None, size + 2)

    def _get_font(self, size):
        if size not in self.fonts:
            self.fonts[size] = self._build_font(size)
        return self.fonts[size]

    def measure(self, text, size):
        return self._get_font(size).size(str(text))

    def line_height(self, size):
        return self._get_font(size).get_linesize()

    def _split_long_token(self, token, size, max_width):
        pieces = []
        current = ""
        for char in token:
            candidate = current + char
            if current and self.measure(candidate, size)[0] > max_width:
                pieces.append(current)
                current = char
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces or [token]

    def wrap_lines(self, text, size, max_width):
        text = str(text)
        if max_width <= 0:
            return [text]

        wrapped = []
        for paragraph in text.splitlines() or [text]:
            if not paragraph:
                wrapped.append("")
                continue

            words = paragraph.split(" ")
            current = ""
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if self.measure(candidate, size)[0] <= max_width:
                    current = candidate
                    continue

                if current:
                    wrapped.append(current)
                    current = ""

                if self.measure(word, size)[0] <= max_width:
                    current = word
                else:
                    wrapped.extend(self._split_long_token(word, size, max_width))

            if current:
                wrapped.append(current)

        return wrapped or [""]

    def render(self, text, size, color, aa=True):
        key = (text, size, color, aa)
        if key not in self.cache:
            self.cache[key] = self._get_font(size).render(text, aa, color)
        return self.cache[key]