import numpy as np
class GeneticOptimization:
    def __init__(self, objectiveFunction, rng, generateIndividual, mutation, crossover, *args, **kwargs):
        self.ObjectiveFunction = objectiveFunction
        self.args = args
        self.ObjectiveFunctionHelper = kwargs['helper']
        self.rng = rng
        self.Mutation = mutation
        self.GenerateIndividual = generateIndividual
        self.Crossover = crossover
        self.fun_count = 0
        self.fitness_progression = []
        self.elite_progression = []


    def TournamentSelection(self, size, population, fitness):
        indexes = self.rng.permutation(len(population))[:size]
        mini = np.argmin(fitness[indexes])
        return population[indexes[mini]]
    
    def GeneratePopulation(self, size):
        return np.array( [ self.GenerateIndividual(self.rng, *self.args) for i in range(size)])
    
    def Objective(self, X):
        self.fun_count += 1
        return self.ObjectiveFunction(X, self.ObjectiveFunctionHelper)
    
    def GetOffspring(self, crossover_rate, tournament_size):
        offSprings = []
        for p1 in self.population:
            if self.rng.random() < crossover_rate:
                p2 = self.TournamentSelection(tournament_size, self.population, self.fitness)
                spring = self.Crossover(p1, p2, self.rng, *self.args)
                offSprings.append(spring)
            else:
                offSprings.append(p1.copy())
        return np.array(offSprings)
    
    def MutateOffSprings(self, pop, mutation_rate):
        for i, x in enumerate(pop):
            if self.rng.random() < mutation_rate:
                self.Mutation(pop[i], self.rng, *self.args)


    def solve(self, population_size, generations, crossover_rate, mutation_rate, tournament_size):
        self.fun_count = 0
        self.fitness_progression = []
        self.elite_progression = []
        self.population = self.GeneratePopulation(population_size)
        self.fitness = np.array( list(self.Objective(x) for x in self.population))
        
        min = np.argmin(self.fitness)
        self.elite = self.population[min]
        self.elite_val = self.fitness[min]
        self.fitness_progression.append(self.elite_val)
        self.elite_progression.append(self.elite)
        for i in range(generations):
            offSprings = self.GetOffspring(crossover_rate, tournament_size)
            self.MutateOffSprings(offSprings, mutation_rate)
            self.population = offSprings
            self.fitness = np.array(list(self.Objective(x) for x in self.population))
        
            min = np.argmin(self.fitness)
            if self.fitness[min] < self.elite_val:    
                self.elite = self.population[min]
                self.elite_val = self.fitness[min]
            self.fitness_progression.append(self.elite_val)
            self.elite_progression.append(self.elite)
        
        dic = dict()
        dic['X'] = self.elite
        dic['fun'] = self.elite_val
        dic['fun_call'] = self.fun_count
        dic['progress_fit'] = self.fitness_progression
        dic['pop_progress'] = self.elite_progression
        return dic






