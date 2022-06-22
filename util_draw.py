import math
from random import randint
from tkinter import Image
from PIL import Image
from aggdraw import Draw, Brush, Path, Pen
import PIL.ImageDraw
import time

class HexGeneratorQRS(object):
    CONST = math.sqrt(3) / 2
    CONST_2 = 1.5
    def __init__(self, edge_length, x_length, y_length):
        self.edge_length = edge_length
        self.y_length = y_length
        self.x_length = x_length

    def __call__(self, q, r, s):
        x = self.edge_length * self.CONST * (q - s) + self.x_length / 2
        y = -self.edge_length * self.CONST_2 * r + self.y_length / 2
        for angle in range(30, 390, 60):
            x += math.cos(math.radians(angle)) * self.edge_length
            y += math.sin(math.radians(angle)) * self.edge_length
            yield x
            yield y

class CityGeneratorQRS(object):
    CONST = math.sqrt(3) / 2
    CONST_2 = 1.5

    def __init__(self, edge_length, x_length, y_length):
        self.edge_length = edge_length
        self.y_length = y_length
        self.x_length = x_length

    def to_canonical(self, q, r, s):
        x = self.edge_length * self.CONST * (q - s) + self.x_length / 2
        y = -self.edge_length * self.CONST_2 * r + self.y_length / 2
        return x, y

    def __call__(self, q, r, s):
        x, y = self.to_canonical(q, r, s)
        yield x
        yield y
        y += self.edge_length * 2
        yield x
        yield y

class StratGeneratorQRS(object):
    CONST = math.sqrt(3) / 2
    CONST_2 = 1.5

    def __init__(self, edge_length, x_length, y_length):
        self.edge_length = edge_length
        self.y_length = y_length
        self.x_length = x_length

    def to_canonical(self, q, r, s):
        x = self.edge_length * self.CONST * (q - s) + self.x_length / 2
        y = -self.edge_length * self.CONST_2 * r + self.y_length / 2
        return x, y

    def __call__(self, q, r, s):
        x, y = self.to_canonical(q, r, s)
        x, y = x - self.CONST * self.edge_length, y + self.edge_length
        yield x
        yield y
        x += self.edge_length * 2 * self.CONST
        yield x
        yield y

class DrawingGenerator(object):
    CONST = math.sqrt(3) / 2
    CONST_2 = 1.5

    def __init__(self, edge_length, x_length, y_length):
        self.edge_length = edge_length
        self.x_length = x_length
        self.y_length = y_length
        self.curr_x = 0
        self.curr_y = 0
        self.hex_list = {}
        x, y = 0, 0
        for angle in range(30, 390, 60):
            x = math.cos(math.radians(angle)) * self.edge_length
            y = math.sin(math.radians(angle)) * self.edge_length
            self.hex_list[angle] = (x, y)

    def update_coordinates(self, q, r, s):
        self.curr_x = self.edge_length * self.CONST * (q - s) + self.x_length / 2
        self.curr_y = -self.edge_length * self.CONST_2 * r + self.y_length / 2
        #return self.curr_x, self.curr_y

    def hex(self):
        x, y = self.curr_x, self.curr_y
        for angle in range(30, 390, 60):
            x += self.hex_list[angle][0]
            y += self.hex_list[angle][1]
            yield x
            yield y
        
    def v_line(self):
        x, y = self.curr_x, self.curr_y
        yield x
        yield y
        y += self.edge_length * 2
        yield x
        yield y

    def h_line(self):
        x, y = self.curr_x, self.curr_y
        x, y = x - self.CONST * self.edge_length, y + self.edge_length
        yield x
        yield y
        x += self.edge_length * 2 * self.CONST
        yield x
        yield y

def draw_grid(yields: dict, fw: dict, strat: dict, cities: dict, N, fname, max_yield=10):
    size = (2 * N + 1) * 60
    image = Image.new("RGB", (size, size), "white")
    draw = Draw(image)
    draw_pil = PIL.ImageDraw.Draw(image)
    #hex_gen_qrs = HexGeneratorQRS(30, 2000, 2000)
    #city_gen = CityGeneratorQRS(30, 2000, 2000)
    #gen_strat = StratGeneratorQRS(30, 2000, 2000)
    gen = DrawingGenerator(30, size, size)
    for pos , yiel in yields.items():
        gen.update_coordinates(*pos)
        color = color_map(yiel, fw[pos], strat[pos], max_yield)
        #hex = hex_gen_qrs(*pos)
        hex = gen.hex()
        draw.polygon(list(hex), Brush(color))
        if cities[pos]:
            city = gen.v_line()
            draw.line(tuple(city), Pen("blue", 5))
        if strat[pos]:
            strategic = gen.h_line()
            draw.line(tuple(strategic), Pen("purple", 5))
    draw.flush()
    image.save(fname)


def color_map(num_yield, has_fw, has_strat, max_yield):
    red, green, blue = (0, 0, 0)
    ratio = num_yield / max_yield
    if has_fw:
        if ratio > 0.5:
            green = 255
            red = 255 - int(255 * ratio)
        else:
            red = 255
            green = int(255 * ratio)
    else:
        red = int(200 * ratio) + 55
        green = int(200 * ratio) + 55
        blue = int(150 * ratio) + 55
    #if has_strat:
    #    blue = 255
    return red, green, blue

def main():
    image = Image.new("RGB", (2000, 2000), "white")
    draw = Draw(image)
    hex_gen_qrs = HexGeneratorQRS(30, 2000, 2000)
    N = 15
    for q in range(-N, N + 1):
        for r in range(max(-N, -q-N), min(N, -q + N) + 1):
            s = -q - r
            color = randint(10, 100), randint(10, 200), randint(10, 300)
            hex = hex_gen_qrs(q, r, s)
            draw.polygon(list(hex), Brush(color))
    draw.flush()
    image.show()

def test_line():
    image = Image.new("RGB", (2000, 2000), "white")
    draw_pil = PIL.ImageDraw.Draw(image)
    draw = Draw(image)
    draw_pil.line((0, 0, 10, 10), fill=128, width=3)
    draw.flush()
    image.show()


if __name__ == "__main__":
    test_line()