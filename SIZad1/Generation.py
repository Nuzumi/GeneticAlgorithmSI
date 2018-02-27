import random
import numpy as np
from Individual import Individual
from operator import attrgetter

class Generation:
    '''Generacja zawiera populacje osobnikow'''
    def __init__(self, population_count, gene_count):
        self.population = []
        self.population_count = population_count
        self.gene_for_individual = gene_count

    '''Tworzenie populacji o danej ilosci osobnikow, ktore posiadaja okreslona ilosc genow'''
    def generate_generation(self):
        for i in range(self.population_count):
            self.population.append(Individual([], self.gene_for_individual))
            self.population[i].generate_random_dna()

    '''Wyliczenie "wartosci" kazdego z osobnikow'''
    def evaluate_all_individuals(self, flow_matrix, distance_matrix):
        for i in self.population:
            i.evaluate_individual(flow_matrix, distance_matrix)

    '''Najlepszy osobnik z populacji'''
    def choose_alpha_individual(self):
        return min(i.value for i in self.population)

    def choose_looser_individual(self):
        return max(i.value for i in self.population)

    def get_average_of_population(self):
        return np.average([i.value for i in self.population])

    def mutate_population(self, p_m):
        for i in self.population:
            if random.uniform(0.0, 1.0) <= p_m:
                i.mutate()

    '''Tworzenie dzieci pod warunkiem wyboru p_x populacji jako rodziców'''
    def make_children(self, px, combine_point_count):
        self.population.sort(key=attrgetter('value'))
        population_shuffle = self.population
        population_shuffle = population_shuffle[0:int(px * len(self.population))]
        rest_population = self.population[0:int((1-px) * len(self.population))+1]
        iterations = len(self.population) - len(rest_population)
        for i in range(0, iterations, 2):
            children = population_shuffle[i].combine_individual(population_shuffle[i + 1], combine_point_count)
            rest_population.append(children[0])
            rest_population.append(children[1])

        return rest_population

