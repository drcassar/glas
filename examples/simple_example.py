'''
This script is a minimal example of how to use GLAS. In this example, we have a
search space with 28 different oxides (stored in the variable compound_list),
and we want to find a glass with the highest silica (SiO2) ratio possible. This
is a trivial problem for humans to solve, but the machine does not know the
trivial answer (that is pure silica) and must search for it.

Apart from the GLAS module, to run this example you will need to have numpy
installed.

'''

import numpy as np
from pprint import pprint
from deap import tools
from glas.search import GLAS


class Searcher(GLAS):
    def __init__(
            self,
            individual_size,
            population_size,
            compound_list,
            hof=None,
    ):
        super().__init__(individual_size, population_size, 'max')
        self.compound_list = compound_list
        self.SiO2_idx = self.compound_list.index('SiO2')
        self.hof = hof

    def fitness_function(self, population):
        '''Computes the fitness function.

        In this problem we want to find a glass with the highest silica ratio
        possible. 

        '''
        pop = np.array(population)
        sum_ = pop.sum(axis=1)
        sum_[sum_ == 0] = 1  # to avoid devision by zero
        SiO2_ratio = pop[:, self.SiO2_idx] / sum_
        return SiO2_ratio

    def eval_population(self, population):
        '''Evaluates the individuals in the population that dont have fitness.

        '''
        invalid_inds = [ind for ind in population if not ind.fitness.valid]
        for ind, fit in zip(invalid_inds, self.fitness_function(invalid_inds)):
            ind.fitness.values = (fit,)  # fitness value must be a tuple

        if self.hof is not None:
            self.hof.update(population)

    def callback(self):
        best_fitness = max([ind.fitness.values[0] for ind in self.population])
        print(
            f'Starting generation {self.generation}. '
            f'Best fitness is {best_fitness:.3f}. '
            f'Maximum fitness = 1.0'
        )
    

# Config

num_generations = 500
population_size = 500
hall_of_fame_size = 10

compound_list = ['Al2O3', 'B2O3', 'BaO', 'Bi2O3', 'CaO', 'CdO', 'Gd2O3', 'GeO2',
                 'K2O', 'La2O3', 'Li2O', 'MgO', 'Na2O', 'Nb2O5', 'P2O5', 'PbO',
                 'Sb2O3', 'SiO2', 'SnO2', 'SrO', 'Ta2O5', 'TeO2', 'TiO2', 'WO3',
                 'Y2O3', 'Yb2O3', 'ZnO', 'ZrO2',]


# Run

hall_of_fame = tools.HallOfFame(hall_of_fame_size)

S = Searcher(
    len(compound_list),
    population_size,
    compound_list,
    hof=hall_of_fame)

S.start()
S.run(num_generations)

print('Showing the best individuals found during the search')
for n, ind in enumerate(S.hof):
    print(f'Position {n+1}')
    ind_dict = {comp: value for comp,value in zip(compound_list, ind) if value > 0}
    pprint(ind_dict)
    print()
