'''
This script is a minimal example of how to use GLAS. In this example, we have a
search space with 28 different oxides (see the variable compound_list), and we
want to find a glass with the highest silica (SiO2) ratio possible. This is a
trivial problem for humans to solve, but the machine does not know the trivial
answer (that is pure silica) and must search for it.

Apart from the GLAS module, to run this example you will need to have numpy
installed. You can do this with pip by running 'pip install numpy'

'''
import numpy as np
from pprint import pprint
from deap import tools
from glas import GLAS


class Searcher(GLAS):
    def __init__(
        self,
        individual_size,
        population_size,
        compound_list,
        hof=None,
    ):
        super().__init__(
            individual_size,
            population_size,
            optimization_goal='max',
        )
        self.compound_list = compound_list
        self.SiO2_idx = self.compound_list.index('SiO2')
        self.hof = hof

    def fitness_function(self, population):
        '''Computes the fitness score.

        In this problem we want to find a glass with the highest silica ratio
        possible. 

        '''
        pop = np.array(population)
        sum_ = pop.sum(axis=1)
        sum_[sum_ == 0] = 1  # to avoid devision by zero
        SiO2_ratio = pop[:, self.SiO2_idx] / sum_
        return SiO2_ratio

    def eval_population(self, population):
        '''Evaluates the individuals in the population that don't have fitness.

        '''
        invalid_inds = [ind for ind in population if not ind.fitness.valid]
        for ind, fit in zip(invalid_inds, self.fitness_function(invalid_inds)):
            ind.fitness.values = (fit, )  # fitness value must be a tuple

        if self.hof is not None:
            self.hof.update(population)

    def callback(self):
        '''What to print at the start of each generation (starting from the 2nd)

        '''
        best_fitness = max([ind.fitness.values[0] for ind in self.population])
        print(
            'Finished generation {0}. '.format(str(self.generation).zfill(3)),
            f'Best fitness is {best_fitness:.3f}. '
            f'Maximum fitness = 1.0')


# Paremeters of the inverse design search

num_generations = 500
population_size = 500
hall_of_fame_size = 10

compound_list = [
    'Al2O3',
    'B2O3',
    'BaO',
    'Bi2O3',
    'CaO',
    'CdO',
    'Gd2O3',
    'GeO2',
    'K2O',
    'La2O3',
    'Li2O',
    'MgO',
    'Na2O',
    'Nb2O5',
    'P2O5',
    'PbO',
    'Sb2O3',
    'SiO2',
    'SnO2',
    'SrO',
    'Ta2O5',
    'TeO2',
    'TiO2',
    'WO3',
    'Y2O3',
    'Yb2O3',
    'ZnO',
    'ZrO2',
]

# Run

hall_of_fame = tools.HallOfFame(hall_of_fame_size)

S = Searcher(
    len(compound_list),
    population_size,
    compound_list,
    hof=hall_of_fame,
)

S.start()
S.run(num_generations)

print(f'The {hall_of_fame_size} best individual(s) found during the search '
      '(composition in mol%)\n')

for n, ind in enumerate(S.hof):
    print(f'Position {n+1}')
    sum_ = 100 / sum(ind)
    ind_dict = {
        comp: round(value * sum_, 2)
        for comp, value in zip(compound_list, ind) if value > 0
    }
    pprint(ind_dict)
    print()
