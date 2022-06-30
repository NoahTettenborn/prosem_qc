import math
from random import randint
from tkinter import Image
from PIL import Image, ImageDraw, ImageFont
from aggdraw import Draw, Brush, Font, Pen
import PIL.ImageDraw
import time

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

    def set_coordinates(self, x, y):
        self.curr_x = x
        self.curr_y = y

    def update_coordinates(self, q, r, s):
        self.curr_x = self.edge_length * self.CONST * (q - s) + self.x_length / 2
        self.curr_y = -self.edge_length * self.CONST_2 * r + self.y_length / 2

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

def draw_grid(yields: dict, fw: dict, strat: dict, cities: dict, N, fname, txt, max_yield=10):
    size = (2 * N + 1) * 60
    image = Image.new("RGB", (size, size), "white")
    draw = Draw(image)
    gen = DrawingGenerator(30, size, size)
    for pos , yiel in yields.items():
        gen.update_coordinates(*pos)
        color = color_map(yiel, fw[pos], max_yield)
        hex = gen.hex()
        hex = list(hex)
        draw.polygon(hex, Brush(color))
        if cities[pos]:
            city = gen.v_line()
            draw.line(tuple(city), Pen("rgb(26, 83, 255)", 10))
            draw.polygon(hex, Pen("rgb(26, 83, 255)", 5))
        if strat[pos]:
            strategic = gen.h_line()
            draw.line(tuple(strategic), Pen("purple", 5))
    font = Font((0, 0, 0), "/Library/Fonts/Microsoft/Arial.ttf", 15)
    draw.text((0, 0), txt, font)

    draw.flush()
    image.save(fname)

def draw_weights(weights: dict, cities:dict, strat:dict, N, fname, max_weight):
    size = (2 * N + 1) * 60
    image = Image.new("RGB", (size, size), "white")
    draw = Draw(image)
    gen = DrawingGenerator(30, size, size)
    for pos , weight in weights.items():
        gen.update_coordinates(*pos)
        color = color_map(-weight, has_fw=True, max_yield=max_weight)
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

def draw_explanation(yields: dict, fw: dict, strat: dict, cities: dict, N, fname, max_yield=10):
    size = (2 * N + 1) * 90
    image = Image.new("RGB", (size, size), "white")
    draw = Draw(image)
    gen = DrawingGenerator(30, size, size)
    for pos , yiel in yields.items():
        gen.update_coordinates(*pos)
        color = color_map(yiel, fw[pos], max_yield)
        #hex = hex_gen_qrs(*pos)
        hex = gen.hex()
        hex = list(hex)
        draw.polygon(hex, Brush(color))
        if cities[pos]:
            city = gen.v_line()
            draw.line(tuple(city), Pen("rgb(26, 83, 255)", 10))
            draw.polygon(hex, Pen("rgb(26, 83, 255)", 5))
        if strat[pos]:
            strategic = gen.h_line()
            draw.line(tuple(strategic), Pen("purple", 5))
    
    font = Font((0, 0, 0), "/Library/Fonts/Microsoft/Arial.ttf", 30)
    txt1 = "Grayscale: No freshwater, lighter means higher yields"
    txt2 = "Red to green: Has fresh water, greener means higher yields"
    txt3 = ": Strategic resorce"
    txt4 = ": City"
    gen.set_coordinates(size/20, size/20)
    hex=list(gen.hex())
    draw.polygon(hex, Brush(color_map(5, False, 10)))
    draw.text((size/10, size/20+10), txt1, font)

    gen.set_coordinates(size/20-25, size/10)
    hex=list(gen.hex())
    draw.polygon(hex, Brush(color_map(3, True, 10)))

    gen.set_coordinates(size/20 + 25, size/10)
    hex=list(gen.hex())
    draw.polygon(hex, Brush(color_map(7, True, 10)))
    draw.text((size/10, size/10+10), txt2, font)

    gen.set_coordinates(size/20, 3*size/20)
    strategic = gen.h_line()
    draw.line(tuple(strategic), Pen("purple", 5))
    draw.text((size/10, 3*size/20+10), txt3, font)
    
    gen.set_coordinates(size/20, size/5)
    city=gen.v_line()
    hex = list(gen.hex())
    draw.line(tuple(city), Pen("rgb(26, 83, 255)", 10))
    draw.polygon(hex, Pen("rgb(26, 83, 255)", 5))
    draw.text((size/10, size/5+10), txt4, font)

    draw.flush()
    image.save(fname)

def color_map(num_yield, has_fw, max_yield):
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

if __name__ == "__main__":
    print("Hey what are you doing, don't execute me!")