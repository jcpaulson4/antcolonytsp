#!/usr/bin/python3

from TSPClasses import *
import time


class TSPSolver:
	def __init__(self, gui_view):
		self._scenario = None
		self.view = gui_view

	def setupWithScenario(self, scenario):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>
		results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm
		</returns>
	'''

	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		city_count = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time() - start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation(city_count)
			route = []
			# Now build the route using the random permutation
			for i in range(city_count):
				route.append(cities[perm[i]])
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'], results['total'], results['pruned'] = None, None, None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>
		results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm
		</returns>
	'''

	def greedy(self, time_allowance: float = 60.0) -> dict:
		results: dict = {}
		cities: list[City] = self._scenario.getCities()
		city_count: int = len(cities)
		route: list[City] = [cities[0]]
		start_time: float = time.time()

		while len(route) < city_count and time.time() - start_time < time_allowance:
			city: City = route[-1]
			best_neighbor: Optional[City] = None
			best_cost: float = math.inf
			neighbor: City
			for neighbor in cities:
				if neighbor in route:
					continue
				neighbor_cost: float = city.costTo(neighbor)
				if neighbor_cost < best_cost:
					best_neighbor = neighbor
					best_cost = neighbor_cost
			if best_neighbor is None:
				# No possible neighbors; return None
				break
			route.append(best_neighbor)

		found: bool = len(route) == city_count
		bssf: Optional[TSPSolution] = TSPSolution(route) if found else None

		end_time: float = time.time()
		results['cost']: float = bssf.cost if found else math.inf
		results['time']: float = end_time - start_time
		results['count']: int = 1 if found else 0
		results['soln']: Optional[TSPSolution] = bssf
		results['max'], results['total'], results['pruned'] = None, None, None
		return results

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>
		results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.
		</returns>
	'''

	def branchAndBound(self, time_allowance=60.0) -> dict:
		results: dict = {}
		cities: list[City] = self._scenario.getCities()
		bssf_change_count: int = 0
		max_queue_size: int = 1
		states_created: int = 1
		pruned_states: int = 0
		start_time: float = time.time()

		greedy_soln: dict = self.greedy(time_allowance)
		bssf: Optional[TSPSolution] = greedy_soln['soln']
		if bssf is None:
			default_random_soln: dict = self.defaultRandomTour(time_allowance - (time.time() - start_time))
			bssf = default_random_soln['soln']

		problem: Subproblem = Subproblem(cities=cities)
		pq: PriorityQueue = PriorityQueue()
		pq.push(problem)

		while not pq.is_empty() and time.time() - start_time < time_allowance:
			problem: Subproblem = pq.pop()
			max_queue_size = max(max_queue_size, len(pq))

			if problem.lower_bound >= bssf.cost:
				# Keep dequeueing because priority is different from lower_bound
				continue

			subproblems: list[Subproblem]
			num_pruned: int
			subproblems, num_pruned = problem.expand(bssf.cost)
			pruned_states += num_pruned
			states_created += num_pruned + len(subproblems)
			subproblem: Subproblem
			for subproblem in subproblems:
				if subproblem.is_finished():
					if subproblem.lower_bound < bssf.cost:
						bssf = subproblem.solution(cities)
						bssf_change_count += 1
				else:
					pq.push(subproblem)

		end_time: float = time.time()
		results['cost']: float = bssf.cost
		results['time']: float = end_time - start_time
		results['count']: int = bssf_change_count
		results['soln']: Optional[TSPSolution] = bssf
		results['max']: int = max_queue_size
		results['total']: int = states_created
		results['pruned']: int = pruned_states + len(pq) + 1
		return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>
		results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm
		</returns>
	'''

	def fancy(self, time_allowance=60.0) -> dict:
		pass
