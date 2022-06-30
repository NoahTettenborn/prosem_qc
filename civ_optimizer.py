"""
    This script applies simulated quantum annealing, provided by the Dwave Ocean
    package, to find optimal city locations of a randomly generated map. It uses the
    package util_draw.py to draw the results.
    The hexagonal coordinate system, as well as the accompanying functions are
    inspired by the blog post "Hexagonal Grids" from Red Blob Games at
    https://www.redblobgames.com/grids/hexagons/ .

    Noah Tettenborn, 2022
    
     Sources:

        https://variable-scope.com/posts/hexagon-tilings-with-python
        
"""
from dimod import BinaryQuadraticModel
from dimod import Vartype
import neal
from random import randint
import numpy as np
import util_draw
import time
import pprint
from matplotlib import pyplot as plt
from pathlib import Path

def main():
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.size"] = 22
    N = 15
    yields, has_fresh_water, has_strat, map_index = generate_grid(N, prob=0.3)
    map_data = yields, has_fresh_water, has_strat, map_index
    params = 200, 1, 20, 10, 10 #fw, yield, strat, cost, quad
    num_sweeps=1e5
    #explain(map_data, a1, a2, a3, a4, b1, num_sweeps, N)
    solve(map_data, *params, num_sweeps, N, draw_map=True)

def anylise_algo(map_data, linear, quad):
    yields, has_fresh_water, has_strat, map_index = map_data    
    linear_arr = np.array(list(linear.values()))
    quad_arr = np.zeros((len(map_index), len(map_index)))
    for (i, j), val in quad.items():
        quad_arr[i, j] = val

    f, axs = plt.subplots(3, 3, figsize=(15, 15))
    min_Es = []
    ratios_valid = []
    probs = 0.01*np.linspace(1, 9, 9)
    for i, ax in enumerate(axs.flat):
        print(i, ax)
        prob = probs[i]
        E=calc_Energies(linear_arr, quad_arr, int(1e5), prob)
        mask = E < 1e5
        E= E[mask]
        ratios_valid.append(np.count_nonzero(mask) / mask.size)
        ax.hist(E, bins=50, log=True, histtype="step", label=f"{prob:.1%}")
        if i > 5:
            ax.set_xlabel(r"E")
        if i % 3==0:
            ax.set_ylabel(r"Count")
        ax.legend()
        min_Es.append(np.amin(E))
    plt.savefig("E_dists-mask.pdf")
    plt.clf()
    plt.step(probs, ratios_valid)
    plt.xlabel(r"Probability")
    plt.ylabel(r"Ratio of valid points")
    plt.savefig("ratios_valid.pdf")
    plt.clf()

    plt.step(probs, min_Es, label="Random sampling from Probability")
    sol, E_ann = optimise(linear, quad, int(1e5))
    plt.axhline(E_ann, label="Annealer", color="r")
    plt.xlabel(r"Probability")
    plt.ylabel(r"Minimum Energy")
    plt.legend()
    plt.savefig("min_es-mask.pdf")

def plot_sweeps(map_data, a1, a2, a3, a4, b1, num_sweeps, N):
    sweeps_arr = np.geomspace(1e3, 1e6, 20)
    for num_sweeps in sweeps_arr:
        has_city, E, occupancy = solve(map_data, a1, a2, a3, a4, b1, num_sweeps, N)
        Energies.append(E)
        occupancies.append(occupancy)
    Energies=np.array(Energies)
    occupancies = np.array(occupancies)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.step(sweeps_arr, Energies, label="Energies")
    ax2.step(sweeps_arr, occupancies, label="Occupancies")
    ax2.set_xlabel("Numbers of Sweeps")
    ax1.set_ylabel(r"Energy [a. u.]")
    ax2.set_ylabel(r"Occupancy [\%]")
    ax2.set_xscale("log")
    f.suptitle(r"Energies and Occupancies for Different Numbers of Sweeps")
    plt.savefig("O_E2.pdf")

def explain(map_data, a1, a2, a3, a4, b1, num_sweeps, N):
    """Makes figure with legend explaining the symbols

    Args:
        map_data (list): entries are qrs positions of grid points
        a_i (float): linear coefficients
        b1 (float): quadratic cost
        num_sweeps (int): number of sweeps to use with annealer
        N (int): size of grid
    """
    has_city = {}
    linear_qrs = {}
    yields, has_fresh_water, has_strat, map_index = map_data
    linear, quad=generate_weights(*map_data, N, a_1=a1, a_2=a2, a_3=a3, a_4=a4, b_1=b1)
    sol, E = optimise(linear, quad, num_sweeps)
    for i, pos in enumerate(map_index):
        has_city[pos] = sol[i]
        linear_qrs[pos] = linear[i]
    util_draw.draw_explanation(yields, has_fresh_water, has_strat, has_city, N, "expl.png")


def solve(map_data, a1, a2, a3, a4, b1, num_sweeps, N, draw_map=False):
    has_city = {}
    yields, has_fresh_water, has_strat, map_index = map_data
    linear, quad=generate_weights(*map_data, N, a_1=a1, a_2=a2, a_3=a3, a_4=a4, b_1=b1)
    sol, E = optimise(linear, quad, num_sweeps)
    num_city = 0
    for i, pos in enumerate(map_index):
        if sol[i]:
            num_city +=1
        has_city[pos] = sol[i]
    if draw_map:
        occupancy = draw(map_data, a1, a2, a3, a4, b1, num_sweeps, N, has_city, num_city, E)
    return has_city, E, occupancy

def draw(map_data, a1, a2, a3, a4, b1, num_sweeps, N, has_city, num_city, E):
    yields, has_fresh_water, has_strat, map_index = map_data
    occupancy = num_city/len(map_index)
    fname = f"./vary_N/{N=}/{a1=}_{a2=}_{b1=}.png"
    Path(f"./vary_N/{N=}").mkdir(parents=True, exist_ok=True)
    text = f"""{a1=}, {a2=}, {a3=}, {a4=}, {b1=}, {len(map_index)} tiles, {num_city} cities
            {occupancy:.2%}occupancy, {E=}, num_sweeps={num_sweeps:.2e}."""
    util_draw.draw_grid(yields, has_fresh_water, has_strat, has_city, N, fname, txt=text)
    return occupancy


def generate_grid(N: int, prob: float):
    """Generate the map randomly, fresh water with probability brob

    Args:
        N (int): Size of grid, equal to number of hex in side - 1
        prob (float): Probability of fresh water

    Returns:
        tuple: (yields: dict, has_fw: dict, has_strat: dict, map_index: list) with
            qrs keys. Map index contains all possible positions as qrs tuples.
    """
    map_index = []
    yields = {}
    has_fw = {}
    has_strat = {}
    gen = valueGenerator(N, prob)
    print(f"The generated map contains {gen.num_hex} tiles.")
    count = 0
    for q in range(-N, N + 1):
        for r in range(max(-N, -q-N), min(N, -q + N) + 1):
            s = -q - r
            map_index.append((q, r, s))
            count += 1
            yields[(q, r, s)] = gen.generate_yield(q, r, s)
            has_fw[(q, r, s)] = gen.generate_fw(q, r, s)
            has_strat[(q, r, s)] = gen.generate_strat(q, r, s)
    return yields, has_fw, has_strat, map_index

def generate_weights(yields:dict, has_fw:dict, has_strat: dict, map_index: list, N: int,
                    a_1=200, a_2=1, a_3=20, a_4=10, b_1=10):
    """Generate the weights as outlined in Sec. 1.2 of the paper. Returns two dicts in a tuple,
    containing the linear and quadratic weights, respectively. Keys are Integers.

    Args:
        yields (dict): qrs keys. Total number of yields for each tile
        has_fw (dict): qrs keys. Whether tiles have fresh water.
        has_strat (dict): qrs keys. Whether tiles have strategic resource
        map_index (list): Contains (qrs) positions of all points
        N (int): Size of map, equal to number of side hex - 1
        a_1 (int, optional): Fresh Water multiplier. Defaults to 20000.
        a_2 (int, optional): Yield multiplier. Defaults to 1.
        a_3 (int, optional): Strategic multiplier. Defaults to 20.
        a_4 (int, optional): Constant cost. Defaults to 0.
        b_1 (int, optional): Quadratic cost. Defaults to 0.

    Returns:
        tuple: Two dicts, contain linear/quadratic weights with int keys.

    """
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
        
        linear[i] = -a_1*has_fw[pos] - a_2*sum_yield - a_3*has_strat[pos] + a_4

        for j, pos2 in enumerate(map_index[0:i]):
            diffs = pos[0] - pos2[0], pos[1] - pos2[1], pos[2] - pos2[2]
            dist = max(abs(diff) for diff in diffs)
            quad[(i, j)] = 1e5 if abs(dist) < 4 else b_1
    return linear, quad

def optimise(linear: dict, quad: dict, n):
    """Apply dwave simulated annealing to linear and quadratic weights.
    Returns the state of system as dict with int keys, according to map_index.

    Args:
        linear (dict): Linear weights. Int keys
        quad (dict): Quadratic weights. Int keys

    Returns:
        dict: Best found solution. Int keys
    """
    sampler = neal.SimulatedAnnealingSampler()
    bqm = BinaryQuadraticModel(linear, quad, Vartype.BINARY)
    sampleset = sampler.sample(bqm, num_sweeps=n)
    print(sampleset.first.energy)
    return sampleset.first.sample, sampleset.first.energy

def calc_Energies(linear: np.ndarray, quad: np.ndarray, num: int, prob:float):
    """Randomly creates sample states and returns their energies. 

    Args:
        linear (np.ndarray): linear weights
        quad (np.ndarray): quadratic weight terms. Rows and columns that don't
            correspond to couplings have to be zero.
        num (int): number of samples to generate
        prob (float): Probability of having a city on every tile

    Returns:
        np.ndarray: Energies of samples, shape=(num, 1)
    """
    sample = np.random.binomial(1, prob, size=(num, 19))
    E = np.zeros(sample.shape[0])
    E = sample * linear
    E = np.sum(E, axis=1, dtype=np.float64)
    print(E.shape)
    E = E + np.einsum('kj,ki,ij->k', sample, sample, quad)
    return E


class valueGenerator(object):
    
    def __init__(self, N, prob, seed=None):
        self.rng = np.random.default_rng(seed)
        self.num_hex = 3 * N * (N + 1) + 1
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


if __name__ == "__main__":
    main()