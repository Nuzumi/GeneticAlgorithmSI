import random
import numpy as np
import itertools as it


def roulette(individual_list, weight_sum):
    value = random.uniform(0.0, 1.0) * weight_sum
    for i in individual_list:
        value -= i.value
        if value < 0:
            return i
    return individual_list[len(individual_list) - 1]


def take_random_from_list(list_to_take, count):
    value = []
    random.shuffle(list_to_take)
    for i in range(count):
        value.append(list_to_take[i])

    return value


def load_files_to_genetic_algorithm(count):
    for i in range(count):
        with open('data' + str(i+1) + '.txt', 'r') as f:
            matrix_dimension = int(f.readline())
            flow_matrix = np.zeros(shape=(matrix_dimension, matrix_dimension))
            distance_matrix = np.zeros(shape=(matrix_dimension, matrix_dimension))
            f.readline()
            for row in range(matrix_dimension):
                flow_matrix[row] = np.fromstring(f.readline(), dtype=int, sep=' ')
            GeneticAlgorithm.flow_matrix_list.append(flow_matrix)

            f.readline()
            for row in range(matrix_dimension):
                distance_matrix[row] = np.fromstring(f.readline(), dtype=int, sep=' ')
            GeneticAlgorithm.distance_matrix_list.append(distance_matrix)


def combine_individual(individual1, individual2, combine_point_count):
    gene_count = individual1.gene_count
    if combine_point_count > gene_count - 1:
        combine_point_count = gene_count - 1
    combine_points = take_random_from_list(list(range(gene_count - 2)), combine_point_count)
    combine_points = list(map(lambda x: x + 1, combine_points))
    combine_points.sort()

    new_dna1 = []
    new_dna2 = []
    swap = False
    swap_counter = 0
    for i in range(gene_count):
        if swap:
            new_dna1.append(individual1.dna[i])
            new_dna2.append(individual2.dna[i])
        else:
            new_dna1.append(individual2.dna[i])
            new_dna2.append(individual1.dna[i])
            pass

        if swap_counter < len(combine_points):
            if i == combine_points[swap_counter]:
                swap_counter += 1
                swap = not swap

    individual_list = [Individual(new_dna1, gene_count), Individual(new_dna2, gene_count)]
    for i in individual_list:
        if i.is_dna_incorrect():
            i.repair_dna()
    return individual_list



def make_children(population, px, combine_point_count):
    population_shuffle = population
    random.shuffle(population_shuffle)
    population_shuffle = population_shuffle[0:(px * len(population_shuffle))]
    rest_population = population - population_shuffle
    for i in range(len(population_shuffle)):
        rest_population.append(combine_individual(population_shuffle[i], population_shuffle[i+1], combine_point_count))
        i += 1

    return rest_population


class Individual:

    def __init__(self, dna, gene_count):
        self.dna = dna
        self.gene_count = gene_count
        self.value = None

    def generate_random_dna(self):
        self.dna = list(range(self.gene_count))
        random.shuffle(self.dna)

    '''zamienia 2 geny ze soba miejscami'''
    def mutate(self):
        index_of_genes_to_swap = take_random_from_list(list(range(self.gene_count)), 2)
        gen1 = self.dna[index_of_genes_to_swap[0]]
        self.dna[index_of_genes_to_swap[0]] = self.dna[index_of_genes_to_swap[1]]
        self.dna[index_of_genes_to_swap[1]] = gen1

    def is_dna_incorrect(self):
        correct = list(range(self.gene_count))
        correct.sort()
        dna = self.dna
        dna.sort()
        if correct == dna:
            return False
        return True

    def repair_dna(self):
        counter = list(np.zeros(shape=(1, len(self.dna)))[0])
        for i in self.dna:
            counter[i] += 1

        while counter.count(0) != 0:
            missing_value = counter.index(0)
            overgrow_value = counter.index(2)
            index_of_missing_value = self.dna.index(overgrow_value)
            self.dna[index_of_missing_value] = missing_value
            counter[missing_value] = 1
            counter[overgrow_value] = 1


class GeneticAlgorithm:

    distance_matrix_list = []
    flow_matrix_list = []

    def __init__(self, p_m, p_x, tour, pop_size, gen, combine_point_count, matrix_index):
        self.p_m = p_m
        self.p_x = p_x
        self.tour = tour
        self.pop_size = pop_size
        self.gen = gen
        self.distance_matrix = self.distance_matrix_list[matrix_index]
        self.flow_matrix = self.flow_matrix_list[matrix_index]
        self.population = []
        self.gene_count = len(self.flow_matrix[0])
        self.pop_value_sum = 0
        self.combine_point_count = combine_point_count
        self.best_of_generations = []
        if tour == 0:
            self.select_function = self.roulette_selection()
        else:
            self.select_function = self.tournament_selection(tour)

    def start_evolution(self):
        gen = 1

        for i in range(self.pop_size):
            self.population.append(Individual([], len(self.distance_matrix[0])))
            self.population[i].generate_random_dna()

        while gen != self.gen:
            self.evaluate_all_individual(self)
            self.best_of_generations.append(self.choose_alpha_individual_of_generation())
            self.population = self.select_function
            self.population = make_children(self.population, self.p_x, self.combine_point_count)
            self.mutate_population()
            gen += 1

        print(self.best_of_generations)
        pass

    def choose_alpha_individual_of_generation(self):
        population = self.population
        population.sort(key=lambda x: x.value , reverse=True)
        return population[0]
        
    def mutate_population(self):
        for i in self.population:
            if random.uniform(0.0, 1.0) <= self.p_m:
                i.mutate()
            
    def evaluate_all_individuals(self):
        for i in self.population:
            self.evaluate_individual(i)

    def evaluate_individual(self, individual):
        value = 0
        count = individual.gene_count
        for i in range(count-1):
            for j in range(i + 1, count):
                value += self.flow_matrix[i][j] * self.distance_matrix[individual.dna[i]][individual.dna[j]]
        individual.value = 1/value
        pass

    def roulette_selection(self):
        selected_individuals = []
        weight_sum = sum(i.value for i in self.population)
        for i in range(len(self.population)):
            selected_individuals.append(roulette(self.population, weight_sum))
        return selected_individuals

    def tournament_selection(self, tour):
        selected_individuals = []
        for i in range(len(self.population)):
            population = self.population
            random.shuffle(population)
            candidates = population[:tour]
            selected_individuals.append(candidates.sort(key=lambda x: x.value, reverse=True)[0])
        return selected_individuals

    def brut_force_solution(self):
        best = Individual()
        best_value = 0
        best_solution = []
        for i in it.permutations(range(self.gene_count)):
            best.dna = i
            self.evaluate_individual()
            if best.value < best_value:
                best_solution = best.dna
                best_value = best.value
        print(best_solution)
        print(best_value)



