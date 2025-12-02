#*******************************************************************************
# File Name: main.py
# Course: Comp 3625 - Artificial Intelligence
# Assignment: 1
# Due Date: 2021-10-06
# Made By: Glenn Yeap & Clency Tabe

# Purpose: This file contains the main code for solving the Travelling Salesman Problem (TSP) using a Genetic Algorithm (GA).
#
# Description: This code uses the PyGAD library to solve the TSP. The code uses a custom fitness function, order crossover, and adaptive hybrid mutation.
#              The code also uses the nearest neighbor algorithm to generate initial solutions. The code also uses the evaluation module to measure the distance
#              of the route and plot the route.
# Functions: find_route, fitness_func, nearest_neighbor, order_crossover, adaptive_hybrid_mutation, distance_between
#
#********************************************************************************

import pandas as pd
import numpy as np
import pygad
import time
import matplotlib.pyplot as plt
import evaluation


#*******************************************************************************
# Function Name: find_route
# Purpose: This function finds the best route for the TSP using a Genetic Algorithm (GA).
# Parameters:
#           locations, a pandas DataFrame containing the x-y coordinates for each location in the TSP.
# Return Value: A list containing the best route for the TSP.
#*******************************************************************************
def find_route(locations: pd.DataFrame) -> list:
    # Ensure the starting city is always the first city
    starting_city = locations.index[0]
    locations = locations.reindex([starting_city] + [city for city in locations.index if city != starting_city])

    # GA Configuration
    num_generations = 500  # Increased number of generations for better convergence
    sol_per_pop = 100  # Increased population size
    num_parents_mating = 70  # Increase number of mating parents
    num_genes = len(locations) - 1  # Number of cities (genes) excluding the starting city

    # Initial population using nearest neighbor and random initialization
    initial_population = [nearest_neighbor(locations) for _ in range(sol_per_pop // 2)]
    initial_population += [np.random.permutation(num_genes) + 1 for _ in range(sol_per_pop // 2)]

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        initial_population=initial_population,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        crossover_type=order_crossover,
        mutation_type=adaptive_hybrid_mutation,
        parent_selection_type="tournament",
        keep_elitism=int(0.1 * sol_per_pop),  # Elitism: Top 10% solutions
        crossover_probability=0.85
    )

    ga_instance.run()
    best_solution, _, _ = ga_instance.best_solution()

    # Prepend the starting city to the solution
    best_solution = np.concatenate(([0], best_solution))
    return best_solution.astype(int).tolist()


def fitness_func(ga_instance, solution, solution_idx):
    # Prepend the starting city index (0) to the solution before calculating distance
    complete_route = np.concatenate(([0], solution)).astype(int)
    return -evaluation.measure_distance(tsp, complete_route)  # Minimize distance


#*******************************************************************************
# Function Name: nearest_neighbor
# Purpose: This function generates an initial solution using the nearest neighbor algorithm.
# Parameters:
#           locations, a pandas DataFrame containing the x-y coordinates for each location in the TSP.
# Return Value: A list containing the initial solution generated using the nearest neighbor algorithm.
#*******************************************************************************
def nearest_neighbor(locations: pd.DataFrame) -> list:
    np_locations = locations.to_numpy()
    num_cities = len(np_locations) - 1 # Exclude the starting city
    visited = [False] * (num_cities + 1)
    current_city = 0
    route = []
    visited[current_city] = True

    for _ in range(num_cities):
        nearest_city = None
        min_distance = float('inf')

        for next_city in range(1, num_cities + 1):
            if not visited[next_city]:
                dist = distance_between(np_locations[current_city], np_locations[next_city])
                if dist < min_distance:
                    min_distance = dist
                    nearest_city = next_city

        current_city = nearest_city
        route.append(current_city)
        visited[current_city] = True

    return route


#*******************************************************************************
# Function Name: order_crossover
# Purpose: This function performs order crossover on the parents to generate offspring.
#
# Parameters:
#           parents, a list containing the parents to perform crossover on.
#           offspring_size, a tuple containing the number of offspring to generate.
#           ga_instance, the GA instance object.
# Return Value: A numpy array containing the offspring generated using order crossover.
#*******************************************************************************
def order_crossover(parents, offspring_size, ga_instance):
    offspring = []
    for k in range(offspring_size[0]):
        parent1 = parents[k % len(parents)]
        parent2 = parents[(k + 1) % len(parents)]

        # Perform order crossover
        child = np.full(len(parent1), -1)
        start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
        child[start:end + 1] = parent1[start:end + 1]

        parent2_remaining = [gene for gene in parent2 if gene not in child]
        pointer = 0
        for i in range(len(child)):
            if child[i] == -1:
                child[i] = parent2_remaining[pointer]
                pointer += 1

        offspring.append(child)

    return np.array(offspring)

#*******************************************************************************
# Function Name: adaptive_hybrid_mutation
# Purpose: This function performs adaptive hybrid mutation on the offspring.
#
# Parameters:
#           offspring, a numpy array containing the offspring to perform mutation on.
#           ga_instance, the GA instance object.
# Return Value: A numpy array containing the offspring after mutation.
#*******************************************************************************
def adaptive_hybrid_mutation(offspring, ga_instance):
    generation = ga_instance.generations_completed
    total_generations = ga_instance.num_generations

    mutation_rate = 0.05 + (generation / total_generations) * 0.45

    for idx in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            mutation_type = np.random.choice(["inversion", "scramble", "two_opt"])

            i, j = sorted(np.random.randint(len(offspring[idx]), size=2))  # Calculate i, j once

            if mutation_type == "inversion":
                offspring[idx][i:j + 1] = np.flip(offspring[idx][i:j + 1])

            elif mutation_type == "scramble":
                np.random.shuffle(offspring[idx][i:j + 1])

            elif mutation_type == "two_opt":
                offspring[idx][i:j + 1] = offspring[idx][i:j + 1][::-1]

    return offspring

#*******************************************************************************
# Function Name: distance_between
# Purpose: This function calculates the Euclidean distance between two points.
# Parameters:
#           point_a, a tuple containing the x-y coordinates of the first point.
#           point_b, a tuple containing the x-y coordinates of the second point.
# Return Value: A float value containing the Euclidean distance between the two points.
#*******************************************************************************
def distance_between(point_a, point_b) -> float:
    return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


if __name__ == '__main__':
    # Load TSP data
    tsp = pd.read_csv('./data/10a.csv', index_col=0)  # Update file path

    for i in range(5):
        start_time = time.time()
        route = find_route(tsp)
        elapsed_time = time.time() - start_time

        distance = evaluation.measure_distance(tsp, route)
        print(f"Route {i + 1}: Distance = {distance:.2f}, Time = {elapsed_time:.4f} seconds")

        evaluation.plot_route(tsp, route)
        plt.title(f'Distance: {distance:.2f} in {elapsed_time:.4f} seconds')
        plt.show()
