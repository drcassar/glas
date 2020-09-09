#!/usr/bin/env python3

import random
from abc import ABC, abstractmethod
from deap import base, creator, tools


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
