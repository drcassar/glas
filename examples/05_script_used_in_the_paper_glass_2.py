'''
This script searches for glasses having a high refractive index and low glass
transition temperature. The constraints and objectives in this example were used
to obtain the Glass 2 reported in the paper.

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
from glas.search import GLAS


class Searcher(GLAS):
    def __init__(
        self,
        individual_size,
        population_size,
        compound_list,
        desired_nd,
        desired_Tg,
        glass_formers,
        min_fraction_formers,
        forced_domain={},
        domain_relax=0,
        weight_nd=1,
        weight_Tg=1,
        hof=None,
    ):
        super().__init__(
            individual_size,
            population_size,
            optimization_goal='min',
        )
        self.compound_list = compound_list
        self.desired_nd = desired_nd
        self.desired_Tg = desired_Tg
        self.weight_nd = weight_nd
        self.weight_Tg = weight_Tg
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

        # Glass transition temperature model and information
        self.model_Tg = load_model(
            r'files/model_glass_transition_temperature.h5')
        features_Tg, domain_Tg, x_mean_Tg, x_std_Tg, Tg_mean, Tg_std = \
            pickle.load(open(r'files/Tg.p', "rb"))
        self.x_mean_Tg = x_mean_Tg
        self.x_std_Tg = x_std_Tg
        self.Tg_std = Tg_std
        self.Tg_mean = Tg_mean
        self.domain_Tg = domain_Tg
        self.features_Tg = features_Tg

        # Chemical domain of the problem
        # This is the intersection of the domain of the predictive models
        elemental_domain = {}
        for element in sorted(set(domain_nd) | set(domain_Tg)):
            elemental_domain[element] = [
                max([
                    domain_nd.get(element, [0, 0])[0],
                    domain_Tg.get(element, [0, 0])[0]
                ]),
                min([
                    domain_nd.get(element, [0, 0])[1],
                    domain_Tg.get(element, [0, 0])[1]
                ]),
            ]

        # Relaxing the domain by the domain_relax factor
        for el, dom in elemental_domain.items():
            elemental_domain[el] = [
                elemental_domain[el][0] * (1 - domain_relax),
                min(elemental_domain[el][1] * (1 + domain_relax), 1),
            ]

        # Forcing the domain for specific elements (given by the user)
        for el, dom in forced_domain.items():
            elemental_domain[el] = dom

        # Compound dictionary and element list
        compound_dicts = []
        all_elements = []
        for c in compound_list:
            cdic = parse_formula(c)
            compound_dicts.append(cdic)
            for el in cdic:
                all_elements.append(el)
        all_elements = list(
            sorted(set(all_elements) | set(domain_nd) | set(domain_Tg)))

        # Conversion matrix, necessary to convert compounds to atomic fraction
        conversion_matrix = np.zeros((len(compound_list), len(all_elements)))
        for j in range(len(compound_list)):
            cdic = compound_dicts[j]
            for el in cdic:
                i = all_elements.index(el)
                conversion_matrix[j, i] += cdic[el]

        self.index_nd = [all_elements.index(el) for el in features_nd]
        self.index_Tg = [all_elements.index(el) for el in features_Tg]
        self.index_formers = [compound_list.index(c) for c in glass_formers]

        self.all_elements = all_elements
        self.elemental_domain = elemental_domain
        self.conversion_matrix = conversion_matrix

    def predict_nd(self, x):
        x_scaled = (x - self.x_mean_nd) / self.x_std_nd
        nd_scaled = self.model_nd.predict(x_scaled)
        nd = nd_scaled * self.nd_std + self.nd_mean
        return nd

    def predict_Tg(self, x):
        x_scaled = (x - self.x_mean_Tg) / self.x_std_Tg
        Tg_scaled = self.model_Tg.predict(x_scaled)
        Tg = Tg_scaled * self.Tg_std + self.Tg_mean
        return Tg

    def fitness_function(self, population):
        '''Computes the fitness score.

        See the paper for more details on how the fitness score was computed.

        '''
        pop = np.array(population)
        atomic_array = pop @ self.conversion_matrix
        sum_ = atomic_array.sum(axis=1)
        sum_[sum_ == 0] = 1
        atomic_array /= sum_.reshape(-1, 1)  # normalization

        predicted_nd = self.predict_nd(atomic_array[:,
                                                    self.index_nd]).flatten()
        predicted_Tg = self.predict_Tg(atomic_array[:,
                                                    self.index_Tg]).flatten()

        fitness = (
            self.weight_nd * (predicted_nd - self.desired_nd)**2 + \
            self.weight_Tg * (predicted_Tg - self.desired_Tg)**2 \
        )**(1/2)

        # Glass-former ratio penalty
        total_sum = pop.sum(axis=1)
        total_sum[total_sum == 0] = 1
        former_ratio = pop[:, self.index_formers].sum(axis=1) / total_sum
        logic_former = former_ratio < self.min_fraction_formers
        distance = self.min_fraction_formers - former_ratio[logic_former]
        fitness[logic_former] += (100 * distance)**2

        # Out-of-domain penalty
        distance = np.zeros(len(pop))
        for n, el in enumerate(self.all_elements):
            el_atomic_frac = atomic_array[:, n]
            el_domain = self.elemental_domain.get(el, [0, 0])
            logic1 = el_atomic_frac > el_domain[1]
            distance[logic1] += el_atomic_frac[logic1] - el_domain[1]
            logic2 = el_atomic_frac < el_domain[0]
            distance[logic2] += el_domain[0] - el_atomic_frac[logic2]
        fitness += (100 * distance)**2

        # Final additional penalty
        logic = np.logical_or(logic_former, distance.astype(bool))
        fitness[logic] += 1e2

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
        best_fitness = min([ind.fitness.values[0] for ind in self.population])
        print(
            'Finished generation {0}. '.format(str(self.generation).zfill(3)),
            f'Best fitness is {best_fitness:.3g}. '
            f'Minimum fitness = 0')


# Parameters of the inverse design search

desired_nd = 1.75
desired_Tg = 400 + 273

weight_nd = 20
weight_Tg = 1

num_generations = 5000
population_size = 400
hall_of_fame_size = 10

minimun_fraction_formers = 0.45
glass_formers = [
    'SiO2',
    'B2O3',
    'GeO2',
    'P2O5',
    'TeO2',
]

forced_domain = {
    'B': [0, 0.02],
    'P': [0, 0.03],
}
domain_relax = 0.2

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

# Running the genetic algorithm search

hall_of_fame = tools.HallOfFame(hall_of_fame_size)

S = Searcher(
    len(compound_list),
    population_size,
    compound_list,
    desired_nd,
    desired_Tg,
    glass_formers,
    minimun_fraction_formers,
    forced_domain,
    domain_relax,
    weight_nd,
    weight_Tg,
    hof=hall_of_fame,
)

S.start()
S.run(num_generations)

# Showing the results

print(f'The {hall_of_fame_size} best individual(s) found during the search '
      '(composition in mol%)')

for n, ind in enumerate(S.hof):
    print(f'Position {n+1}')
    sum_ = 100 / sum(ind)
    ind_dict = {
        comp: round(value * sum_, 2)
        for comp, value in zip(compound_list, ind) if value > 0
    }
    pprint(ind_dict)
    print()
