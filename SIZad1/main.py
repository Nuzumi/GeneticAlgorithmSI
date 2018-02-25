import Classes as Cl


def main():
    Cl.load_files_to_genetic_algorithm(1)
    a = Cl.GeneticAlgorithm(0.03, 0.5, 0, 10, 10, 2, 0)
    a.start_evolution()
    '''
    ind.generate_random_dna()
    a = Cl.GeneticAlgorithm(0, 0, 0, 0, 0, 0)
    a.evaluate_individual(ind)
    '''
    pass


if __name__ == "__main__":
    main()
