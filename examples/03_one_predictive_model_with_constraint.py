'''
This script searches for glasses having a high refractive index. The search is
constrained to compositions having a minimum amount of glass-formers.

Apart from the GLAS module, to run this example you will need to have the
following Python modules installed:
    - numpy
    - tensorflow
    - chemparse

You can install these modules using pip by running

'pip install numpy tensorflow chemparse'

'''
import pickle
import numpy as np
from pprint import pprint
from deap import tools
from chemparse import parse_formula
from tensorflow.keras.models import load_model
from glas import GLAS


class Searcher(GLAS):
    def __init__(
        self,
        individual_size,
        population_size,
        compound_list,
        glass_formers,
        min_fraction_formers,
        hof=None,
    ):
        super().__init__(
            individual_size,
            population_size,
            optimization_goal='max',
        )
        self.compound_list = compound_list
        self.min_fraction_formers = min_fraction_formers
        self.hof = hof

        # Refractive index model and information
        self.model_nd = load_model(r'files/model_refractive_index.h5')
        features_nd, domain_nd, x_mean_nd, x_std_nd, nd_mean, nd_std = \
            pickle.load(open(r'files/nd.p', "rb"))
        self.x_mean_nd = x_mean_nd
        self.x_std_nd = x_std_nd
        self.nd_std = nd_std
        self.nd_mean = nd_mean
        self.domain_nd = domain_nd
        self.features_nd = features_nd

        # Compound dictionary and element list
        compound_dicts = []
        all_elements = []
        for c in compound_list:
            cdic = parse_formula(c)
            compound_dicts.append(cdic)
            for el in cdic:
                all_elements.append(el)
        all_elements = list(sorted(set(all_elements) | set(domain_nd)))

        # Conversion matrix, necessary to convert compounds to atomic fraction
        conversion_matrix = np.zeros((len(compound_list), len(all_elements)))
        for j in range(len(compound_list)):
            cdic = compound_dicts[j]
            for el in cdic:
                i = all_elements.index(el)
                conversion_matrix[j, i] += cdic[el]

        self.index_nd = [all_elements.index(el) for el in features_nd]
        self.index_formers = [compound_list.index(c) for c in glass_formers]
        self.all_elements = all_elements
        self.conversion_matrix = conversion_matrix

    def predict_nd(self, x):
        x_scaled = (x - self.x_mean_nd) / self.x_std_nd
        nd_scaled = self.model_nd.predict(x_scaled)
        nd = nd_scaled * self.nd_std + self.nd_mean
        return nd

    def fitness_function(self, population):
        '''Computes the fitness score.

        In this problem we want to find a glass with high refractive index. A
        simple constraint on a minimum amount of glass-formers is considered.

        Glasses with a negative fitness score are those penalized by the
        constraint of the problem.

        '''
        pop = np.array(population)
        atomic_array = pop @ self.conversion_matrix
        sum_ = atomic_array.sum(axis=1)
        sum_[sum_ == 0] = 1
        atomic_array /= sum_.reshape(-1, 1)  # normalization

        predicted_nd = self.predict_nd(atomic_array[:,
                                                    self.index_nd]).flatten()
        fitness = predicted_nd

        # Glass-former ratio penalty
        total_sum = pop.sum(axis=1)
        total_sum[total_sum == 0] = 1  # avoiding division by zero
        former_ratio = pop[:, self.index_formers].sum(axis=1) / total_sum
        logic_former = former_ratio < self.min_fraction_formers
        distance = self.min_fraction_formers - former_ratio[logic_former]
        fitness[logic_former] -= (100 * distance)**2

        return fitness

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
            f'Highest refractive index found is {best_fitness:.3g}. ')


# Paremeters of the inverse design search

num_generations = 500
population_size = 500
hall_of_fame_size = 10

minimun_fraction_formers = 0.6
glass_formers = [
    'SiO2',
    'B2O3',
    'GeO2',
    'P2O5',
    'TeO2',
]

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
    glass_formers,
    minimun_fraction_formers,
    hof=hall_of_fame,
)

S.start()
S.run(num_generations)

print(
    f'The {hall_of_fame_size} best individual(s) found during the search '
    '(composition in mol%)\n'
    '(Please note that negative refractive indices denote that the composition\n'
    'is outside the constraint of the problem)\n')

for n, ind in enumerate(S.hof):
    print(f'Position {n+1}')
    sum_ = 100 / sum(ind)
    ind_dict = {
        comp: round(value * sum_, 2)
        for comp, value in zip(compound_list, ind) if value > 0
    }
    pprint(ind_dict)
    print()
