import random
import numpy as np
import Classes as Cl
from random import randint
import copy


class Individual:

    ''' Osobnik ma swoje DNA, ilość genów oraz value, które oznacza "droge" '''
    def __init__(self, dna, gene_count):
        self.dna = dna
        self.gene_count = gene_count
        self.value = None

    '''Randomowe wyznaczenie lokacji/ dna'''
    def generate_random_dna(self):
        '''dna to lista lokacji -> tyle ile moze byc genow'''
        self.dna = list(range(self.gene_count))
        random.shuffle(self.dna)

    '''Wwyliczenie drogi w zaleznossci od nadrzednych danych'''
    def evaluate_individual(self, flow_matrix, distance_matrix):
        value = 0
        count = self.gene_count
        for i in range(count):
            for j in range(i, count):
                x = distance_matrix[i][j]
                y = flow_matrix[self.dna[i]][self.dna[j]]
                value += (2 * x * y)
        self.value = value
        pass

    '''zamienia 2 geny ze soba miejscami'''
    def mutate(self):
        index_of_genes_to_swap = Cl.take_random_from_list(list(range(self.gene_count)), 2)
        gen1 = self.dna[index_of_genes_to_swap[0]]
        self.dna[index_of_genes_to_swap[0]] = self.dna[index_of_genes_to_swap[1]]
        self.dna[index_of_genes_to_swap[1]] = gen1

    def is_dna_incorrect(self):
        correct = list(range(self.gene_count))
        correct.sort()
        dna = copy.copy(self.dna)
        dna.sort()
        if correct == dna:
            return False
        return True

    def repair_dna(self):
        counter = list(np.zeros(shape=(1, len(self.dna)))[0])

        dna = self.dna[:]

        validate_dna = set(range(self.gene_count))
        unique_dna = set(dna)

        missing_values = validate_dna - unique_dna

        missing_values = list(missing_values)

        for i in range(len(self.dna)):
            counter[self.dna[i]] += 1
            if counter[self.dna[i]] == 2:
                index = randint(0, len(missing_values) - 1)
                random_value = missing_values[index]
                dna[i] = random_value
                missing_values.remove(random_value)
                counter[self.dna[i]] -= 1

        self.dna = dna

    '''Krzyzowanie osobnika z innym w zaleznosci od punktow krzyzowania -> zrobic wybieralne krzyzowanie'''
    def combine_individual(self, individual2, combine_point_count):
        gene_count = self.gene_count
        if combine_point_count > gene_count - 1:
            combine_point_count = gene_count - 1
        combine_points = Cl.take_random_from_list(list(range(gene_count - 2)), combine_point_count)
        combine_points = list(map(lambda x: x + 1, combine_points))
        combine_points.sort()

        'new_dna1 = self.dna[0:int(self.gene_count / 2)] + individual2.dna[0:int(self.gene_count / 2):]'
        'new_dna2 = individual2.dna[0:int(self.gene_count / 2)] + self.dna[int(self.gene_count / 2):]'

        new_dna1 = []
        new_dna2 = []

        swap = False

        swap_counter = 0

        for i in range(gene_count):
            if swap:
                new_dna1.append(self.dna[i])
                new_dna2.append(individual2.dna[i])
            else:
                new_dna1.append(individual2.dna[i])
                new_dna2.append(self.dna[i])

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
