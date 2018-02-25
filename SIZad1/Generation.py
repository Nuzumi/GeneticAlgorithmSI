import random
from Individual import Individual


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
        population = self.population
        population.sort(key=lambda x: x.value, reverse=True)
        return population[0]

    def mutate_population(self, p_m):
        for i in self.population:
            if random.uniform(0.0, 1.0) <= p_m:
                i.mutate()

    '''Tworzenie dzieci pod warunkiem wyboru p_x populacji jako rodziców'''
    def make_children(self, px, combine_point_count):
        population_shuffle = self.population
        random.shuffle(population_shuffle)
        population_shuffle = population_shuffle[0:int(px * len(population_shuffle))]
        rest_population = self.population[int(px * len(population_shuffle)):]
        for i in range(len(population_shuffle) - 1):
            children = population_shuffle[i].combine_individual(population_shuffle[i + 1], combine_point_count)
            rest_population.append(children[0])
            rest_population.append(children[1])
            i += 1

        return rest_population

