import random
import numpy as np
import Classes as Cl

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
                y = flow_matrix[self.dna[i]-1][self.dna[j]-1]
                value += 2 * x * y
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

    '''Krzyzowanie osobnika z innym w zaleznosci od punktow krzyzowania -> zrobic wybieralne krzyzowanie'''
    def combine_individual(self, individual2, combine_point_count):
        gene_count = self.gene_count
        if combine_point_count > gene_count - 1:
            combine_point_count = gene_count - 1
        combine_points = Cl.take_random_from_list(list(range(gene_count - 2)), combine_point_count)
        combine_points = list(map(lambda x: x + 1, combine_points))
        combine_points.sort()

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
