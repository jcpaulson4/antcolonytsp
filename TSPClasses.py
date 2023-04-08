#!/usr/bin/python3

import math
from typing import Optional
import heapq
import numpy as np
import random

DEPTH_WEIGHT: int = 312


class TSPSolution:
    def __init__(self, listOfCities):
        self.route = listOfCities
        self.cost = self._costOfRoute()

    def _costOfRoute(self):
        cost = 0
        last = self.route[0]
        for city in self.route[1:]:
            cost += last.costTo(city)
            last = city
        cost += self.route[-1].costTo(self.route[0])
        return cost

    def enumerateEdges(self):
        elist = []
        c1 = self.route[0]
        for c2 in self.route[1:]:
            dist = c1.costTo(c2)
            if dist == np.inf:
                return None
            elist.append((c1, c2, int(math.ceil(dist))))
            c1 = c2
        dist = self.route[-1].costTo(self.route[0])
        if dist == np.inf:
            return None
        elist.append((self.route[-1], self.route[0], int(math.ceil(dist))))
        return elist


def nameForInt(num):
    if num == 0:
        return ''
    elif num <= 26:
        return chr(ord('A') + num - 1)
    else:
        return nameForInt((num - 1) // 26) + nameForInt((num - 1) % 26 + 1)


class Scenario:
    HARD_MODE_FRACTION_TO_REMOVE = 0.20  # Remove 20% of the edges

    def __init__(self, city_locations, difficulty, rand_seed):
        self._difficulty = difficulty

        if difficulty == "Normal" or difficulty == "Hard":
            self._cities = [City(pt.x(), pt.y(), \
                                 random.uniform(0.0, 1.0) \
                                 ) for pt in city_locations]
        elif difficulty == "Hard (Deterministic)":
            random.seed(rand_seed)
            self._cities = [City(pt.x(), pt.y(),
                                 random.uniform(0.0, 1.0)
                                 ) for pt in city_locations]
        else:
            self._cities = [City(pt.x(), pt.y()) for pt in city_locations]

        num = 0
        for city in self._cities:
            city.setScenario(self)
            city.setIndexAndName(num, nameForInt(num + 1))
            num += 1

        # Assume all edges exists except self-edges
        ncities = len(self._cities)
        self._edge_exists = (np.ones((ncities, ncities)) - np.diag(np.ones((ncities)))) > 0

        if difficulty == "Hard":
            self.thinEdges()
        elif difficulty == "Hard (Deterministic)":
            self.thinEdges(deterministic=True)

    def getCities(self):
        return self._cities

    def randperm(self, n):
        perm = np.arange(n)
        for i in range(n):
            randind = random.randint(i, n - 1)
            save = perm[i]
            perm[i] = perm[randind]
            perm[randind] = save
        return perm

    def thinEdges(self, deterministic=False):
        ncities = len(self._cities)
        edge_count = ncities * (ncities - 1)  # can't have self-edge
        num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE * edge_count)

        can_delete = self._edge_exists.copy()

        # Set aside a route to ensure at least one tour exists
        route_keep = np.random.permutation(ncities)
        if deterministic:
            route_keep = self.randperm(ncities)
        for i in range(ncities):
            can_delete[route_keep[i], route_keep[(i + 1) % ncities]] = False

        # Now remove edges until
        while num_to_remove > 0:
            if deterministic:
                src = random.randint(0, ncities - 1)
                dst = random.randint(0, ncities - 1)
            else:
                src = np.random.randint(ncities)
                dst = np.random.randint(ncities)
            if self._edge_exists[src, dst] and can_delete[src, dst]:
                self._edge_exists[src, dst] = False
                num_to_remove -= 1


class City:
    def __init__(self, x, y, elevation=0.0):
        self._x = x
        self._y = y
        self._elevation = elevation
        self._scenario = None
        self._index = -1
        self._name = None

    def setIndexAndName(self, index, name):
        self._index = index
        self._name = name

    def setScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		How much does it cost to get from this city to the destination?
		Note that this is an asymmetric cost function.
		 
		In advanced mode, it returns infinity when there is no connection.
		</summary> '''
    MAP_SCALE = 1000.0

    def costTo(self, other_city):

        assert type(other_city) == City

        # In hard mode, remove edges; this slows down the calculation...
        # Use this in all difficulties, it ensures INF for self-edge
        if not self._scenario._edge_exists[self._index, other_city._index]:
            return np.inf

        # Euclidean Distance
        cost = math.sqrt((other_city._x - self._x) ** 2 +
                         (other_city._y - self._y) ** 2)

        # For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
        if not self._scenario._difficulty == 'Easy':
            cost += (other_city._elevation - self._elevation)
            if cost < 0.0:
                cost = 0.0

        return int(math.ceil(cost * self.MAP_SCALE))


class Subproblem:
    def __init__(self, cities: Optional[list[City]] = None,
                 matrix: Optional[list[list[float]]] = None, path: Optional[list[int]] = None,
                 lower_bound: Optional[float] = None, depth: int = 0):
        # Initialize based on list of cities
        if matrix is None:
            self.city_count: int = len(cities)
            self.matrix: list[list[float]] = [[0] * self.city_count for _ in range(self.city_count)]
            self.path: list[int] = [0]
            self.lower_bound: float = 0
            self.depth: int = depth

            # Initialize and reduce matrix
            for i in range(self.city_count):
                for j in range(self.city_count):
                    self.matrix[i][j] = cities[i].costTo(cities[j])
            self.reduce()
        # Initialize based on previous subproblem
        elif cities is None:
            self.city_count: int = len(matrix)
            self.matrix = [row[:] for row in matrix]
            self.path: list[int] = [item for item in path]
            self.lower_bound: float = lower_bound
            self.depth: int = depth

    def reduce(self):
        reduction_cost: float = 0
        # Reduce rows
        for i in range(self.city_count):
            if i in self.path and i != self.path[-1]:
                continue
            minimum: float = min(self.matrix[i])
            if minimum > 0:
                reduction_cost += minimum
                for j in range(self.city_count):
                    self.matrix[i][j] -= minimum
        # Reduce columns
        for j in range(self.city_count):
            if j in self.path and j != self.path[0]:
                continue
            minimum: float = min([self.matrix[i][j] for i in range(self.city_count)])
            if minimum > 0:
                reduction_cost += minimum
                for i in range(self.city_count):
                    self.matrix[i][j] -= minimum
        self.lower_bound += reduction_cost

    def add_to_path(self, row: int, col: int):
        self.lower_bound += self.matrix[row][col]
        self.path.append(col)
        for i in range(self.city_count):
            self.matrix[row][i] = math.inf
            self.matrix[i][col] = math.inf

    def __gt__(self, other) -> bool:
        return self.get_priority() > other.get_priority()

    def __lt__(self, other) -> bool:
        return self.get_priority() < other.get_priority()

    def get_priority(self) -> float:
        return self.lower_bound - (self.depth * DEPTH_WEIGHT)

    def is_finished(self) -> bool:
        return self.city_count == len(self.path)

    def solution(self, cities: list[City]) -> Optional[TSPSolution]:
        if not self.is_finished():
            return None
        return TSPSolution([cities[i] for i in self.path])

    def expand(self, bssf_cost: float) -> tuple:
        results = []
        num_pruned: int = 0
        for i in range(self.city_count):
            if i in self.path:
                continue
            sub = Subproblem(matrix=self.matrix, path=self.path, lower_bound=self.lower_bound, depth=self.depth + 1)
            sub.add_to_path(self.path[-1], i)
            sub.reduce()
            if sub.lower_bound < bssf_cost:
                results.append(sub)
            else:
                num_pruned += 1
        return results, num_pruned


class PriorityQueue:
    def __init__(self):
        self.queue: list = []

    def push(self, sub: Subproblem):
        heapq.heappush(self.queue, (sub.get_priority(), sub))

    def pop(self) -> Subproblem:
        _, sub = heapq.heappop(self.queue)
        return sub

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def __len__(self) -> int:
        return len(self.queue)
