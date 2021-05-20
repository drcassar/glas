import random
import operator
from multiprocess import Pool  # https://stackoverflow.com/a/65001152
from abc import ABC, abstractmethod
from types import SimpleNamespace
from pprint import pprint

import numpy as np
from deap import base, creator, tools, gp
from chemparse import parse_formula
from mendeleev import element


class BaseGLAS(ABC):
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
    mutation_prob = 0.2
    mating_prob = 0.5

    def __init__(self, population_size):
        super().__init__()
        self.population_size = population_size

    @abstractmethod
    def eval_population(self, population):
        pass

    @abstractmethod
    def create_toolbox(self):
        pass

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
                offspring = self.toolbox.select(self.population,
                                                k=len(self.population))
                offspring = list(map(self.toolbox.clone, offspring))

                # Apply crossover on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.mating_prob:
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


class BaseChemistryGLAS(BaseGLAS):

    gene_crossover_prob = 0.5
    tournament_size = 3
    gene_mut_prob = 0.05
    minimum_composition = 0
    maximum_composition = 100

    def __init__(
            self,
            individual_size,
            population_size,
            compound_list,
            elements,
            optimization_goal='min',
    ): 
        super().__init__(population_size)

        self.individual_size = individual_size
        self.optimization_goal = optimization_goal

        # Compound dictionary and element list
        compound_dicts = []
        all_elements_from_comp = []
        for comp in compound_list:
            cdic = parse_formula(comp)
            compound_dicts.append(cdic)
            for el in cdic:
                all_elements_from_comp.append(el)
        all_elements = list(sorted(set(all_elements_from_comp) | set(elements)))

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
        self.conversion_matrix = conversion_matrix
        self.compound_list = compound_list

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


class GlassSearcher(BaseChemistryGLAS):

    base_penalty = 1000
    report_frequency = 50

    def __init__(self, config, design, constraints={}):
        self.config = SimpleNamespace(**config)
        self.design = design

        self.hof = tools.HallOfFame(self.config.hall_of_fame_size)

        super().__init__(
            individual_size=len(self.config.compound_list),
            population_size=self.config.population_size,
            compound_list=self.config.compound_list,
            optimization_goal='min',
            elements=[element(n).symbol for n in range(1, 93)],
        )

        domain_list = []
        self.models = {}
        self.report = {}
        for ID in design:
            cls = design[ID]['class'](
                all_elements=self.all_elements,
                compound_list=self.config.compound_list,
                **design[ID].get('info', {}),
            )
            if design[ID]['use_for_optimization']:
                domain_list.append(cls.get_domain())
                self.models[ID] = cls
            else:
                self.report[ID] = cls

        self.penalties = []
        for ID in constraints:
            cls = constraints[ID]['class']
            self.penalties.append(
                cls(
                    constraints[ID]['config'],
                    domain_list=domain_list,
                    all_elements=self.all_elements,
                    **config,
                )
            )

    def fitness_function(self, population):
        pop_dict = {
            'population_array': np.array(population),
            'population_weight': self.population_to_weight(population),
            'atomic_array': self.population_to_atomic_array(population),
        }

        fitness = np.zeros(len(population))

        for ID in self.models:
            y_pred = self.models[ID].predict(pop_dict)
            conf = self.design[ID]['config']

            is_higher_than_max = np.greater(y_pred, conf['max'])
            is_lower_than_min = np.less(y_pred, conf['min'])
            wr = is_within_range = np.logical_not(
                np.logical_or(is_higher_than_max, is_lower_than_min)
            )

            x0 = conf['min']
            y0 = conf['weight'] if conf['objective'] == 'maximize' else 0

            x1 = conf['max']
            y1 = conf['weight'] if conf['objective'] == 'minimize' else 0

            m = (y1 - y0) / (x1 - x0)

            fitness[is_within_range] = fitness[is_within_range] + \
                m * (y_pred[is_within_range] - x0) + y0

            fitness[~wr] = fitness[~wr] + self.base_penalty + \
                is_higher_than_max[~wr] * (y_pred[~wr] - conf['max'])**2 + \
                is_lower_than_min[~wr] * (conf['min'] - y_pred[~wr])**2
                
        for cls in self.penalties:
            penalty = cls.compute(pop_dict, self.base_penalty)
            fitness += penalty

        return fitness

    def eval_population(self, population):
        invalid_inds = [ind for ind in population if not ind.fitness.valid]
        for ind, fit in zip(invalid_inds, self.fitness_function(invalid_inds)):
            ind.fitness.values = (fit, )  # fitness value must be a tuple
        if self.hof is not None:
            self.hof.update(population)

    def report_dict(self, individual, verbose=True):
        report_dict = {}
        ind_array = np.array(individual).reshape(1,-1)

        pop_dict = {
            'population_array': ind_array,
            'population_weight': self.population_to_weight(ind_array),
            'atomic_array': self.population_to_atomic_array(ind_array),
        }

        if verbose:
            pprint(self.ind_to_dict(individual))

        if verbose:
            print()
            print('Predicted properties of this individual:')

        for ID in self.models:
            y_pred = self.models[ID].predict(pop_dict)[0]
            report_dict[ID] = y_pred
            if verbose:
                print(f'{self.design[ID]["name"]} = {y_pred:.3f} '
                      f'{self.design[ID].get("unit", "")}')

        for ID in self.report:
            y_pred = self.report[ID].predict(pop_dict)[0]
            within_domain = self.report[ID].is_within_domain(pop_dict)[0]
            if within_domain:
                report_dict[ID] = y_pred
                if verbose:
                    print(f'{self.design[ID]["name"]} = {y_pred:.3f} '
                        f'{self.design[ID].get("unit", "")}')
            elif verbose:
                print(f'{self.design[ID]["name"]} = Out of domain')

        if verbose:
            print()
            print()

        return report_dict

    def callback(self):
        best_fitness = min([ind.fitness.values[0] for ind in self.population])
        print(
            'Finished generation {0}. '.format(str(self.generation).zfill(3)),
            f'Best fitness is {best_fitness:.3g}. '
        )

        if self.generation % self.report_frequency == 0:
            if best_fitness < self.base_penalty:
                best_ind = tools.selBest(self.population, 1)[0]
                print('\nBest individual in this population (in mol%):')
                self.report_dict(best_ind, verbose=True)


class GPSR(BaseGLAS):

    chance_select_ind_lower_lenght = 0.65  # min 0.5, max 1
    tournament_fitness_first = True
    tree_max_depth = 17
    tournment_size = 3

    def __init__(self, config, num_jobs=1):
        self.config = SimpleNamespace(**config)
        self.hof = tools.HallOfFame(self.config.hall_of_fame_size)

        super().__init__(self.config.population_size)

        self.pset = self.config.pset
        self.optimization_goal = self.config.optimization_goal
        self.num_jobs = num_jobs

    def create_toolbox(self):

        # https://deap.readthedocs.io/en/master/api/gp.html#deap.gp.PrimitiveTree
        # https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
        # https://deap.readthedocs.io/en/master/tutorials/basic/part4.html

        if self.optimization_goal.lower() in ['max', 'maximum']:
            creator.create("Fitness", base.Fitness, weights=(1.0, ))
        elif self.optimization_goal.lower() in ['min', 'minimum']:
            creator.create("Fitness", base.Fitness, weights=(-1.0, ))
        else:
            raise ValueError('Invalid optimization_goal value.')

        creator.create("Individual", gp.PrimitiveTree,
                        fitness=creator.Fitness)

        toolbox = base.Toolbox()

        if self.num_jobs != 1:
            pool = Pool(processes=self.num_jobs)
            toolbox.register("map", pool.map)

        toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=1,
            max_=2,
        )

        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            toolbox.expr,
        )

        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
        )

        toolbox.register("compile", gp.compile, pset=self.pset)

        def evalfun(individual, x, y, fitfun, pset):
            compiled_fun = gp.compile(individual, pset)
            y_pred = np.array([compiled_fun(*a) for a in x])
            if all(np.isfinite(y_pred)):
                fitness = (fitfun(y, y_pred), )
            else:
                fitness = (np.nan, )
            return fitness

        toolbox.register(
            "evaluate",
            evalfun,
            x=self.config.x,
            y=self.config.y,
            fitfun=self.config.fitfun,
            pset=self.pset,
        )

        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register(
            "mutate",
            gp.mutUniform,
            expr=toolbox.expr_mut,
            pset=self.pset,
        )
        toolbox.decorate(
            "mutate",
            gp.staticLimit(
                key=operator.attrgetter("height"),
                max_value=self.tree_max_depth
            )
        )

        toolbox.register(
            "select",
            tools.selDoubleTournament,
            fitness_size=self.tournment_size,
            parsimony_size=self.chance_select_ind_lower_lenght * 2,
            fitness_first=self.tournament_fitness_first, 
        )

        toolbox.register("mate", gp.cxOnePoint)
        toolbox.decorate(
            "mate",
            gp.staticLimit(
                key=operator.attrgetter("height"),
                max_value=self.tree_max_depth
            )
        )

        return toolbox

    def eval_population(self, population):
        invalid_inds = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_inds)
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit
        if self.hof is not None:
            self.hof.update(population)

    def eval_individual(self, individual, x):
        compiled_fun = self.toolbox.compile(expr=individual)
        y_pred = np.array([compiled_fun(*a) for a in x])
        return y_pred
