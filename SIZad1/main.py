import Classes as Cl


def main():
    Cl.load_files_to_genetic_algorithm(1)
    ind = Cl.Individual([0, 5, 1, 2, 4, 3], 6)
    ind2 = Cl.Individual([4, 1, 2, 0, 3, 5], 6)
    '''
    ind.generate_random_dna()
    a = Cl.GeneticAlgorithm(0, 0, 0, 0, 0, 0)
    a.evaluate_individual(ind)
    '''
    pass


if __name__ == "__main__":
    main()
