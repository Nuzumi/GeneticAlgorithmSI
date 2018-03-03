import random
import numpy as np
from Generation import Generation as Gen
from Individual import Individual as Ind
import itertools as it
from operator import attrgetter
import copy

def load_files_to_genetic_algorithm(count):
    for i in range(count):
        with open('data' + str(i + 1) + '.txt', 'r') as f:
            matrix_dimension = int(f.readline())
            flow_matrix = np.zeros(shape=(matrix_dimension, matrix_dimension))
            distance_matrix = np.zeros(shape=(matrix_dimension, matrix_dimension))

            f.readline()
            for row in range(matrix_dimension):
                distance_matrix[row] = np.fromstring(f.readline(), dtype=int, sep=' ')
            GeneticAlgorithm.distance_matrix_list.append(distance_matrix)

            f.readline()
            for row in range(matrix_dimension):
                flow_matrix[row] = np.fromstring(f.readline(), dtype=int, sep=' ')
            GeneticAlgorithm.flow_matrix_list.append(flow_matrix)


def take_random_from_list(list_to_take, count):
    value = []
    random.shuffle(list_to_take)
    for i in range(count):
        value.append(list_to_take[i])

    return value


class GeneticAlgorithm:

    distance_matrix_list = []
    flow_matrix_list = []
    first_generation = None

    def __init__(self, p_m, p_x, tour, pop_size, generations_count, combine_point_count, matrix_index):
        self.p_m = p_m
        self.p_x = p_x
        self.tour = tour
        self.pop_size = pop_size
        self.generations_count = generations_count
        '''Utworzenie macierzy, pozniejsze ich wypelnienie'''
        self.distance_matrix = self.distance_matrix_list[matrix_index]
        self.flow_matrix = self.flow_matrix_list[matrix_index]
        '''bedziemy operowac na liscie generacji, ktora ma swoja populacje osobnikow'''
        self.generations = []
        self.gene_count = len(self.flow_matrix[0])
        self.pop_value_sum = 0
        self.combine_point_count = combine_point_count
        self.best_of_generations = []
        self.average_of_generations = []
        self.worst_of_generations = []
        '''Zalezny wybor metody selekcji'''
        if tour == 0:
            self.select_function = self.roulette_selection
        else:
            self.select_function = self.tournament_selection

    def __str__(self):
        return 'PM: ' + str(self.p_m) + ' PX: ' + str(self.p_x) + ' Tur: ' + str(self.tour) + ' Populacja: ' + str(self.pop_size) + ' Ilość generacji: ' + str(self.generations_count) + ' Punkty krzyżowania: ' + str(self.combine_point_count)


    def start_evolution(self):
        gen = 0
        while gen != self.generations_count:
            '''Tworze kolejna generacje, dodaje ja do listy'''

            if gen == 0:
                generation = Gen(self.pop_size, self.gene_count)
                '''Pierwsza generacje losuje'''
                if GeneticAlgorithm.first_generation is None:
                    generation.generate_generation()
                    GeneticAlgorithm.first_generation = copy.deepcopy(generation)
                else:
                    generation = copy.deepcopy(GeneticAlgorithm.first_generation)
                '''Tworze kolejna generacje, dodaje ja do listy'''
                self.generations.append(generation)
                '''Ewaluuje w niej wszystkich osobnikow'''
                self.generations[gen].evaluate_all_individuals(self.flow_matrix, self.distance_matrix)
            else:
                '''Przepisuje populacje "wyzej" i na niej dzialam, potem zmieniam ja na jej dzieci'''
                generation = self.generations[gen-1]
                self.generations.append(generation)

                self.generations[gen].evaluate_all_individuals(self.flow_matrix, self.distance_matrix)
                '''Podmieniam te generacje na tylko wybranych prawdopodobnych do zostania rodzicem'''
                self.generations[gen].population = self.select_function(self.generations[gen].population, self.tour)
                '''Podmieniam generacje na jej dzieci'''

                self.generations[gen].evaluate_all_individuals(self.flow_matrix, self.distance_matrix)
                self.generations[gen].population.sort(key=attrgetter('value'))
                self.generations[gen].population = self.generations[gen].make_children(self.p_x, self.combine_point_count)
                '''Mutuje'''
                self.generations[gen].mutate_population(self.p_m)
                '''Ewaluuje w niej wszystkich osobnikow'''
                self.generations[gen].evaluate_all_individuals(self.flow_matrix, self.distance_matrix)

            '''Biore sobie najlepszego z generacji'''
            best_alpha_of_gen = self.generations[gen].choose_alpha_individual()
            worst_of_gen = self.generations[gen].choose_looser_individual()
            self.best_of_generations.append(best_alpha_of_gen)
            self.worst_of_generations.append(worst_of_gen)
            self.average_of_generations.append(self.generations[gen].get_average_of_population())
            '''
            print(gen)
            print(len(self.generations[gen].population))
            '''
            gen += 1
        pass

    def roulette_selection(self, population, dummy):
        selected_individuals = []
        weight_sum = sum(i.value for i in population)
        for i in range(len(population)):
            selected_individuals.append(self.roulette(population, weight_sum))
        return selected_individuals

    def tournament_selection(self, population, tour):
        selected_individuals = []
        for i in range(len(population)):
            population = population
            random.shuffle(population)
            candidates = population[:tour]
            candidate = min(candidates, key=attrgetter('value'))
            selected_individuals.append(candidate)
        return selected_individuals

    @staticmethod
    def roulette(individual_list, weight_sum):
        value = random.uniform(0.0, 1.0) * weight_sum
        for i in individual_list:
            value -= i.value
            if value < 0:
                return i
        return individual_list[len(individual_list) - 1]

    def brut_force_solution(self):
        best = Ind([], self.gene_count)
        best_value = 0
        best_solution = []
        count = 1
        for i in it.permutations(range(self.gene_count)):
            best.dna = i
            best.evaluate_individual(self.flow_matrix, self.distance_matrix)
            print(count)
            if best.value > best_value:
                best_solution = best.dna
                best_value = best.value
            count += 1
        print(best_solution)
        print(best_value)

