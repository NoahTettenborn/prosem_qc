"""
    This script uses... Sources:

        https://variable-scope.com/posts/hexagon-tilings-with-python
        https://www.redblobgames.com/grids/hexagons/#basics
"""
from sys import getsizeof
import neal
from random import randint
from random import getrandbits
from PIL import Image, ImageDraw
import numpy as np
from PIL import ImageFont
import util_draw

def main():
    N = 4
    sampler = neal.SimulatedAnnealingSampler()
    yields, has_fresh_water, has_strat, has_city, map_index= generate_grid(N)

    print(getsizeof(yields), getsizeof(has_fresh_water), getsizeof(has_strat), getsizeof(has_city))
    generate_weights(yields, has_fresh_water, has_strat, has_city, map_index, N)
    util_draw.draw_grid(yields, has_fresh_water, has_strat, has_city)
    #draw("./v1.png", yields)


def generate_grid(N):
    map_index = []
    yields = {}
    has_fw = {}
    has_strat = {}
    has_city = {}
    gen = valueGenerator(N)
    print(gen.num_hex)
    count = 0
    for q in range(-N, N + 1):
        for r in range(max(-N, -q-N), min(N, -q + N) + 1):
            s = -q - r
            map_index.append((q, r, s))
            count += 1
            yields[(q, r, s)] = gen.generate_yield(q, r, s)
            has_fw[(q, r, s)] = gen.generate_fw(q, r, s)
            has_strat[(q, r, s)] = gen.generate_strat(q, r, s)

            has_city[(q, r, s)] = gen.generate_city(q, r, s)
    print(map_index)
    return yields, has_fw, has_strat, has_city, map_index

def generate_weights(yields:dict, has_fw:dict, has_strat: dict, has_city: dict, map_index: np.ndarray, N):
    linear = {}
    quad = {}
    for i, pos in enumerate(map_index):
        q, r, s = pos
        sum_yield = yields[pos]
        print(pos)
        if q < N:
            if s > -N:
                sum_yield += yields[(q+1, r, s-1)]
            if r > -N:
                sum_yield += yields[(q+1, r-1, s)]
        if q > -N:
            print("here")
            if s < N:
                sum_yield += yields[(q-1, r, s+1)]
            if r < N:
                sum_yield += yields[(q-1, r+1, s)]
        if r < N & s > -N:
            sum_yield += yields[(q, r+1, s-1)]
        if r > -N & s < N:
            sum_yield += yields[(q, r-1, s+1)]
        

        linear[i] = -sum_yield - 200*has_fw[pos] - 20*has_strat[pos]
        print(f"{linear=}")
        for j, pos2 in enumerate(map_index[0:i-1]):
            print(f"{i=},    {j=}")
            quad[(i, j)] = 2000 if abs(i-j) < 5 else 0
            print(f"{quad=}")
    return linear, quad

class valueGenerator(object):
    
    def __init__(self, N, seed=42):
        rng = np.random.default_rng(seed)
        self.num_hex = 3 * (N + 1) ** 2 - 2
        self.choice_arr = rng.choice([True, False], size=3* self.num_hex, p=[0.2, 0.8])
        self.count = -1

    def generate_yield(self, q, r, s):
        return randint(0, 10)

    def generate_fw(self, q, r, s):
        self.count += 1
        return self.choice_arr[self.count]

    def generate_strat(self, q, r, s):
        self.count += 1
        return self.choice_arr[self.count]

    def generate_city(self, q, r, s):
        self.count += 1
        return self.choice_arr[self.count]

def initialize_vars(N: int):
    cities_neal = {}
    for q in range(-N, N + 1):
        for r in range(max(-N, -q-N), min(N, -q + N) + 1):
            s = -q - r


if __name__ == "__main__":
    main()