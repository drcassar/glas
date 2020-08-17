import random
from deap import base, creator, tools
from abc import ABC, abstractmethod



class GLAS(ABC):
    def __init__(
            self,
            individual_size,
            population_size,
            # fitness_function,
            # constraints=None,
            optimization_goal='min',
    ):
        super().__init__()

        self.individual_size = individual_size
        self.population_size = population_size
        # self.fitness_function = fitness_function
        self.constraints = None
        self.optimization_goal = optimization_goal

        self.tournament_size = 3
        self.minimum_composition = 0
        self.maximum_composition = 100
        self.gene_mut_prob = 0.05
        self.gene_crossover_prob = 0.5
        self.crossover_prob = 0.5
        self.mutation_prob = 0.2

    # @abstractmethod
    # def fitness_function(self):
    #     pass

    @abstractmethod
    def eval_population(self, population):
        pass

    def create_toolbox(self):

        if self.optimization_goal.lower() in ['max', 'maximum']:
            creator.create("Fitness", base.Fitness, weights=(1.0, ))
        elif self.optimization_goal.lower() in ['min', 'minimum']:
            creator.create("Fitness", base.Fitness, weights=(-1.0, ))
        else:
            raise ValueError('Invalide optimization_goal value.')

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

        # toolbox.register("evaluate", self.fitness_function)

        if self.constraints:
            for constraint_dic in self.constraints:
                toolbox.decorate(
                    'evaluate',
                    tools.DeltaPenalty(
                        constraint_dic['check'],
                        constraint_dic['penalty'],
                        constraint_dic.get('distance', None),
                    ),
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
        # fitness = list(map(self.toolbox.evaluate, self.population))
        # fitness = self.fitness_function()

        # for ind, fit in zip(self.population, fitness):
        #     ind.fitness.values = fit

        self.generation = 1

    def callback(self):
        print(f'Starting generation {self.generation}')
        
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

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            self.eval_population(offspring)

            # fitness = self.fitness_function()
            # for ind, fit in zip(invalid_ind, fitness):
            #     ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            self.population[:] = offspring

            self.generation += 1
