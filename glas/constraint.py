from abc import ABC, abstractmethod
import numpy as np


class Constraint(ABC):
    @abstractmethod
    def compute(self):
        pass


class ConstraintGlassFormers(Constraint):
    def __init__(self, config, compound_list, **kwargs):
        super().__init__()
        self.__glass_formers = config['glass_formers']
        self.__min_fraction = config['minimum_fraction_formers']
        self.__index_formers = [
            compound_list.index(comp) for comp in config['glass_formers']
        ]

    def compute(self, population_dict, base_penalty):
        population_array = population_dict['population_array']
        total_sum = population_array.sum(axis=1)
        total_sum[total_sum == 0] = 1

        former_ratio = \
            population_array[:, self.__index_formers].sum(axis=1) / total_sum
        logic_former = former_ratio < self.__min_fraction

        penalty = np.zeros(population_dict['population_array'].shape[0])
        distance = self.__min_fraction - former_ratio[logic_former]
        penalty[logic_former] = (100 * distance)**2 + base_penalty

        return penalty


class ConstraintElements(Constraint):
    def __init__(self, config, compound_list, domain_list, all_elements,
                 **kwargs):
        super().__init__()

        elemental_domain = {}
        for element_ in all_elements:
            minimum = max([d.get(element_, [0 ,0])[0] for d in domain_list])
            maximum = min([d.get(element_, [0 ,0])[1] for d in domain_list])
            elemental_domain[element_] = [minimum, maximum]

        # Relaxing the domain by the domain_relax factor
        for el, dom in elemental_domain.items():
            elemental_domain[el] = [
                elemental_domain[el][0] * (1 - config['domain_relax']),
                min(elemental_domain[el][1] * (1 + config['domain_relax']), 1),
            ]

        # Forcing the domain for specific elements (given by the user)
        for el, dom in config['forced_domain'].items():
            elemental_domain[el] = dom

        self.__elemental_domain = elemental_domain

    def compute(self, population_dict, base_penalty):
        distance = np.zeros(population_dict['atomic_array'].shape[0])

        for n, el in enumerate(self.__elemental_domain):
            el_atomic_frac = population_dict['atomic_array'][:, n]
            el_domain = self.__elemental_domain.get(el, [0, 0])

            logic1 = el_atomic_frac > el_domain[1]
            distance[logic1] += el_atomic_frac[logic1] - el_domain[1]

            logic2 = el_atomic_frac < el_domain[0]
            distance[logic2] += el_domain[0] - el_atomic_frac[logic2]

        logic = distance > 0
        distance[logic] = (100 * distance[logic])**2 + base_penalty
        penalty = distance

        return penalty


class ConstraintCompounds(Constraint):
    def __init__(self, config, compound_list, **kwargs):
        super().__init__()
        self.__forced_domain = config['forced_compound_domain']
        self.__compound_list = compound_list

    def compute(self, population_dict, base_penalty):
        population_array = population_dict['population_array']
        total_sum = population_array.sum(axis=1)
        total_sum[total_sum == 0] = 1

        pop_norm = population_array / total_sum.reshape(-1,1)
        distance = np.zeros(population_array.shape[0])

        for compound in self.__forced_domain:
            idx = self.__compound_list.index(compound)
            domain = self.__forced_domain[compound]

            logic1 = pop_norm[:, idx] > domain[1]
            distance[logic1] += pop_norm[:, idx][logic1] - domain[1]

            logic2 = pop_norm[:, idx] < domain[0]
            distance[logic2] += domain[0] - pop_norm[:, idx][logic2]

        logic = distance > 0
        distance[logic] = (100 * distance[logic])**2 + base_penalty
        penalty = distance

        return penalty
