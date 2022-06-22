"""
    This script uses... Sources:

        https://variable-scope.com/posts/hexagon-tilings-with-python
        https://www.redblobgames.com/grids/hexagons/#basics
"""
from dimod import BinaryQuadraticModel
from dimod import Vartype
import neal
from random import randint
import numpy as np
import util_draw
import time

def main():
    N = 7
    sampler = neal.SimulatedAnnealingSampler()
    yields, has_fresh_water, has_strat, map_index = generate_grid(N, prob=0.3)
    has_city = {}
    map_data = yields, has_fresh_water, has_strat, map_index
    clin=100
    cquad = 100
    fname = f"./vary_cost/{clin}_{cquad}/{time.asctime()[11:-5]}.png"
    linear, quad=generate_weights(*map_data, N, cost_lin=clin, cost_quad=cquad)
    sol = optimise(linear, quad)
    for i, pos in enumerate(map_index):
        has_city[pos] = sol[i]
    util_draw.draw_grid(yields, has_fresh_water, has_strat, has_city, N, fname)

#    linear, quad = generate_weights(*map_data, N)
#    sol = optimise(linear, quad)
#    for i, pos in enumerate(map_index):
#        has_city[pos] = sol[i]
#    
#    util_draw.draw_grid(yields, has_fresh_water, has_strat, has_city, N, fname)
#    #draw("./v1.png", yields)


def generate_grid(N, prob):
    map_index = []
    yields = {}
    has_fw = {}
    has_strat = {}
    has_city = {}
    gen = valueGenerator(N, prob)
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

            #has_city[(q, r, s)] = gen.generate_city(q, r, s)
    return yields, has_fw, has_strat, map_index

def generate_weights(yields:dict, has_fw:dict, has_strat: dict, map_index: list, N: int,
                    yield_weight=1, water_weight=1, cost_lin=0, cost_quad=0):
    linear = {}
    quad = {}
    for i, pos in enumerate(map_index):
        q, r, s = pos
        sum_yield = yields[pos]
        if q < N:
            if s > -N:
                sum_yield += yields[(q+1, r, s-1)]
            if r > -N:
                sum_yield += yields[(q+1, r-1, s)]
        if q > -N:
            if s < N:
                sum_yield += yields[(q-1, r, s+1)]
            if r < N:
                sum_yield += yields[(q-1, r+1, s)]
        if r < N and s > -N:
            sum_yield += yields[(q, r+1, s-1)]
        if r > -N and s < N:
            sum_yield += yields[(q, r-1, s+1)]
        
        linear[i] = -sum_yield - 20000*has_fw[pos] - 20*has_strat[pos] + cost_lin

        #print(f"{linear=}")
        for j, pos2 in enumerate(map_index[0:i]):
            #print(f"{i=},    {j=}")
            diffs = pos[0] - pos2[0], pos[1] - pos2[1], pos[2] - pos2[2]
            dist = max(abs(diff) for diff in diffs)
            quad[(i, j)] = 2000000 if abs(dist) < 4 else cost_quad
            #print(f"{pos=}, {pos2=}, {quad[(i, j)]:>5}")
    return linear, quad

def optimise(linear, quad):
    sampler = neal.SimulatedAnnealingSampler()
    bqm = BinaryQuadraticModel(linear, quad, Vartype.BINARY)
    sampleset = sampler.sample(bqm, num_sweeps=500000)
    print(sampleset.first.energy)
    return sampleset.first.sample

class valueGenerator(object):
    
    def __init__(self, N, prob, seed=None):
        self.rng = np.random.default_rng(seed)
        self.num_hex = 3 * (N + 1) ** 2 - 2
        self.choice_arr = self.rng.choice([True, False], size=3* self.num_hex, p=[prob, 1-prob])
        self.count = -1

    def generate_yield(self, q, r, s):
        return randint(0, 10)

    def generate_fw(self, q, r, s):
        self.count += 1
        return self.choice_arr[self.count]

    def generate_strat(self, q, r, s):
        self.count += 1
        return self.rng.uniform() < 0.1

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