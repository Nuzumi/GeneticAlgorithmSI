import Classes as Cl


def main():
    Cl.load_files_to_genetic_algorithm(1)
    ind = Cl.Individual([], 12)
    ind2 = Cl.Individual([], 12)
    ind.generate_random_dna()
    ind2.generate_random_dna()
    a = Cl.GeneticAlgorithm(0, 0, 0, 0, 0, 0)
    a.evaluate_individual(ind)
    a.evaluate_individual(ind2)
    print(ind.value)
    print(ind2.value)
    '''
    ind.generate_random_dna()
    a = Cl.GeneticAlgorithm(0, 0, 0, 0, 0, 0)
    a.evaluate_individual(ind)
    '''
    pass


if __name__ == "__main__":
    main()
