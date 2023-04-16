from TSPSolver import *
from TSPClasses import *

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

MAX_SEED = 1000

def newPoints(seed: int, npoints: int):
    random.seed(seed)

    SCALE = 1.0
    data_range = {'x': [-1.5 * SCALE, 1.5 * SCALE], 'y': [-SCALE, SCALE]}

    ptlist = []
    xr = data_range['x']
    yr = data_range['y']
    while len(ptlist) < npoints:
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 1.0)
        if True:
            xval = xr[0] + (xr[1] - xr[0]) * x
            yval = yr[0] + (yr[1] - yr[0]) * y
            ptlist.append(QPointF(xval, yval))
    return ptlist

def run(size: int, algorithm: str):
    seed = random.randint(0, MAX_SEED-1)

    points = newPoints(seed, size)  # uses current rand seed
    diff = "Hard (Deterministic)"
    scenario = Scenario(city_locations=points, difficulty=diff, rand_seed=seed)
    genParams = {'size': size, 'seed': seed, 'diff': diff}
    solver = TSPSolver(None)
    solver.setupWithScenario(scenario)

    print(f"Running {algorithm} with size {size} and seed {seed}.")
    solution: dict
    if algorithm == "greedy":
        solution = solver.greedy()
    elif algorithm == "2opt":
        solution = solver.two_opt()
    elif algorithm == "mod 2opt":
        solution = solver.two_opt_mod()
    elif algorithm == "b&b":
        solution = solver.branchAndBound()
    else:
        solution = solver.defaultRandomTour()
    print(f"    {round(solution['time'], 3)}, {solution['cost']}")

for any_algorithm in ("greedy", "b&b", "mod 2opt", "2opt"):
    for any_size in (15, 30, 60, 100, 200):
        for _ in range(5):
            run(any_size, any_algorithm)

