#!/usr/bin/env python3

import random
import pickle
import numpy as np
import os
from abc import ABC, abstractmethod
from pathlib import Path
from deap import base, creator, tools
from chemparse import parse_formula
from tensorflow.keras.models import load_model
from mendeleev import element


base_path = Path(os.path.dirname(__file__)) / 'models'

class GLAS(ABC):
    '''Base class for using GLAS.

    This is just a base class to build your optimization design. The recommended
    way to use is to create a new class that is an instance of this one. At
    least one method must be defined in this new class, which is
    'eval_population'. It is also recommended to create a 'fitness_function'
    method for readability, but it is not required.

    You can check some examples of using GLAS at
    https://github.com/drcassar/glas/tree/master/examples

    Parameters
    ----------
    individual_size : int
        How many "genes" an individual has. This is the number of compounds
        in the chemical space.

    population_size : int
        How many individuals are present in the population. Increasing this
        value can help the problem converge faster, but requires more
        computational resources.

    optimization_goal : 'min' or 'max'
        Configure the type of optimization problem being solved. If 'min' then
        individuals with lower fitness score have a higher chance of surviving,
        otherwise if 'max',

    '''
    tournament_size = 3
    minimum_composition = 0
    maximum_composition = 100
    gene_mut_prob = 0.05
    gene_crossover_prob = 0.5
    crossover_prob = 0.5
    mutation_prob = 0.2

    def __init__(
            self,
            individual_size,
            population_size,
            optimization_goal='min',
    ):
        super().__init__()

        self.individual_size = individual_size
        self.population_size = population_size
        self.optimization_goal = optimization_goal

    @abstractmethod
    def eval_population(self, population):
        pass

    def create_toolbox(self):
        if self.optimization_goal.lower() in ['max', 'maximum']:
            creator.create("Fitness", base.Fitness, weights=(1.0, ))
        elif self.optimization_goal.lower() in ['min', 'minimum']:
            creator.create("Fitness", base.Fitness, weights=(-1.0, ))
        else:
            raise ValueError('Invalid optimization_goal value.')

        creator.create("Individual", list, fitness=creator.Fitness)

        toolbox = base.Toolbox()

        toolbox.register(
            "attr_int",
            random.randint,
            self.minimum_composition,
            self.maximum_composition,
        )

        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_int,
            n=self.individual_size,
        )

        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
        )

        toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=self.minimum_composition,
            up=self.maximum_composition,
            indpb=self.gene_mut_prob,
        )

        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.tournament_size,
        )

        toolbox.register(
            "mate",
            tools.cxUniform,
            indpb=self.gene_crossover_prob,
        )

        return toolbox

    def start(self):
        self.toolbox = self.create_toolbox()
        self.population = self.toolbox.population(n=self.population_size)
        self.eval_population(self.population)
        self.generation = 1

    def callback(self):
        print(f'Finished generation {self.generation}')
        
    def run(self, num_generations=100):
        try:
            for _ in range(num_generations):

                self.callback()

                # Select the next generation individuals
                offspring = self.toolbox.select(self.population, k=len(self.population))
                offspring = list(map(self.toolbox.clone, offspring))

                # Apply crossover on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.crossover_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Apply mutation on the offspring
                for mutant in offspring:
                    if random.random() < self.mutation_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                self.eval_population(offspring)
                self.population[:] = offspring

                self.generation += 1
        except KeyboardInterrupt:
            print('----- Interrupted by the user -----')


class GLAS_property:
    def __init__(self):

        # Refractive index model and information
        self.model_nd = load_model(base_path / 'model_refractive_index.h5')
        features_nd, domain_nd, x_mean_nd, x_std_nd, nd_mean, nd_std = \
            pickle.load(open(base_path / 'nd.p', "rb"))
        self.x_mean_nd = x_mean_nd
        self.x_std_nd = x_std_nd
        self.nd_std = nd_std
        self.nd_mean = nd_mean
        self.domain_nd = domain_nd
        self.features_nd = features_nd

        # Glass transition temperature model and information
        self.model_Tg = load_model(base_path /
                                   'model_glass_transition_temperature.h5')
        features_Tg, domain_Tg, x_mean_Tg, x_std_Tg, Tg_mean, Tg_std = \
            pickle.load(open(base_path / 'Tg.p', "rb"))
        self.x_mean_Tg = x_mean_Tg
        self.x_std_Tg = x_std_Tg
        self.Tg_std = Tg_std
        self.Tg_mean = Tg_mean
        self.domain_Tg = domain_Tg
        self.features_Tg = features_Tg

        # Abbe number model and information
        self.model_abbe = load_model(base_path / 'model_abbe.h5')
        features_abbe, domain_abbe, x_mean_abbe, x_std_abbe, abbe_mean, abbe_std = \
            pickle.load(open(base_path / 'abbe.p', "rb"))
        self.x_mean_abbe = x_mean_abbe
        self.x_std_abbe = x_std_abbe
        self.abbe_std = abbe_std
        self.abbe_mean = abbe_mean
        self.domain_abbe = domain_abbe
        self.features_abbe = features_abbe
    
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

    def predict_abbe(self, x):
        x_scaled = (x - self.x_mean_abbe) / self.x_std_abbe
        abbe_scaled = self.model_abbe.predict(x_scaled)
        abbe = abbe_scaled * self.abbe_std + self.abbe_mean
        return abbe


class GLAS_chemistry:
    def __init__(
            self,
            compound_list,
            domain_list,
            domain_relax=0,
            forced_domain={},
            additional_elements=[],
    ): 

        # Chemical domain of the problem
        # This is the intersection of the domain of the predictive models

        all_elements = set().union(*(d.keys() for d in domain_list))
        elemental_domain = {}
        for element_ in sorted(all_elements):
            minimum = max([d.get(element_, [0 ,0])[0] for d in domain_list])
            maximum = min([d.get(element_, [0 ,0])[1] for d in domain_list])
            elemental_domain[element_] = [minimum, maximum]

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
        all_elements_from_comp = []
        for comp in compound_list:
            cdic = parse_formula(comp)
            compound_dicts.append(cdic)
            for el in cdic:
                all_elements_from_comp.append(el)
        all_elements = list(sorted(set(all_elements_from_comp) | all_elements | set(additional_elements)))

        # Conversion matrix, necessary to convert compounds to atomic fraction
        conversion_matrix = np.zeros((len(compound_list), len(all_elements)))
        for j in range(len(compound_list)):
            cdic = compound_dicts[j]
            for el in cdic:
                i = all_elements.index(el)
                conversion_matrix[j, i] += cdic[el]

        # Molar mass, necessary to convert compounds in mol to wt
        self.molar_mass = [
            sum([element(el).mass * num for el, num in parse_formula(comp).items()])
            for comp in compound_list
        ]
        self.molar_mass = np.diag(self.molar_mass)

        self.all_elements = all_elements
        self.elemental_domain = elemental_domain
        self.conversion_matrix = conversion_matrix
        self.compound_list = compound_list

    def population_to_atomic_array(self, pop):
        atomic_array = pop @ self.conversion_matrix
        sum_ = atomic_array.sum(axis=1)
        sum_[sum_ == 0] = 1
        atomic_array /= sum_.reshape(-1, 1)  # normalization
        return atomic_array

    def population_to_weight(self, pop):
        weight_pct = pop @ self.molar_mass
        sum_ = weight_pct.sum(axis=1)
        sum_[sum_ == 0] = 1
        weight_pct /= sum_.reshape(-1, 1)  # normalization
        return weight_pct

    def ind_to_dict(self, ind):
        sum_ = 100 / sum(ind)
        ind_dict = {
            comp: round(value * sum_, 2)
                for comp, value in zip(self.compound_list, ind) if value > 0
        }
        return ind_dict
