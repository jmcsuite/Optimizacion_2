import numpy as np

class EvolutionaryProgramming:
    def __init__(self, func, bounds, args=(), popsize=30):
        self.func = func
        self.bounds = np.array(bounds)
        self.args = args
        self.popsize = popsize
        self.population = None
        self.sigmas = None
        self.fitness = None
        
    def init_population(self):
        nvar = len(self.bounds)
        self.population = np.zeros((self.popsize, nvar))
        self.sigmas = np.zeros((self.popsize, nvar))
        self.fitness = np.zeros((self.popsize))
        
        for v in range(nvar):
            vmin, vmax = self.bounds[v, 0], self.bounds[v, 1]
            vmut = np.abs(vmax - vmin) / 10
            self.population[:,v] = np.random.uniform(vmin, vmax, (self.popsize))
            self.sigmas[:, v] = np.abs(np.random.normal(loc=0, scale=vmut, size=(self.popsize))) + 0.001
            
        for i in range(self.popsize):
            P = self.population[i,:]
            self.fitness[i] = self.func(P, *self.args)
    
    def mutation(self, alpha=0.2, epsilon=0.001):
        print('in')
        new_population = self.population.copy()
        new_sigmas = self.sigmas.copy()
        nvar = len(self.bounds)
        for v in range(nvar):
            print(new_population[:, v])
            new_sigmas[:, v] = np.maximum(epsilon, new_sigmas[:, v] * (1 + np.random.normal(loc=0, scale=alpha)))
            new_population[:, v] += np.random.normal(loc=0, scale=new_sigmas[:, v])
                                                     
        return new_population
                                                     
                                                      
    def survivor_selection(self, new_population):
        print('survivor')
        two_populations = np.append(self.population, new_population, axis=0)
        print(two_populations)
        two_fitness = np.zeros(two_populations.shape)
        
        for i in range(self.popsize * 2):
            P = two_populations[i,:]
            two_fitness[i] = self.func(P, *self.args)
            
        order = np.argsort(two_fitness)[:self.popsize]
        self.population = two_populations[order]
        self.fitness = two_fitness[order]
        
    
    def solve(self, max_it=300):
        self.init_population()
        elite = self.population[np.argmin(self.fitness)].copy()
        elite_fx = np.min(self.fitness)
        
        nit = 0
        while nit < max_it and elite_fx > 0:
            new_population = self.mutation() 
            self.survivor_selection(new_population)
                                                     
            if(np.min(self.fitness) < elite_fx):
                elite = population[np.argmin(self.fitness)].copy()
                elite_fx = np.min(self.fitness)       
            nit += 1
                                                     
        # P: solution
        # nit: generation count
        # fun: fitness of the fittest
        # nfev: call count to objective function
        return elite, nit, elite_fx, nfev

def evolutionary_programming(func, bounds, args=(), popsize=30):
    print('Evolutionary Programming')
    ep = EvolutionaryProgramming(func, bounds, args, popsize)
    return ep.solve()