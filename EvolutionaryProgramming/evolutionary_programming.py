import numpy as np

class EvolutionaryProgramming:
    def __init__(self, func, bounds, args=(), popsize=30, rng = np.random.default_rng(16072001)):
        self.func = func
        self.bounds = np.array(bounds)
        self.args = args
        self.popsize = popsize
        self.population = None
        self.sigmas = None
        self.fitness = None
        self.rng = rng
        self.nfev = 0
       
    def init_population(self):
        nvar = len(self.bounds)
        self.population = np.zeros((self.popsize, nvar))
        self.sigmas = np.zeros((self.popsize, nvar))
        self.fitness = np.zeros((self.popsize))
        
        for v in range(nvar):
            vmin, vmax = self.bounds[v, 0], self.bounds[v, 1]
            vmut = np.abs(vmax - vmin) / 10
            self.population[:,v] = self.rng.uniform(vmin, vmax, (self.popsize))
            self.sigmas[:, v] = np.abs(self.rng.normal(loc=0, scale=vmut, size=(self.popsize))) + 0.001
        
        self.fitness = self.objective_func(self.population, *self.args)
        order = np.argsort(self.fitness)
        self.fitness = self.fitness[order]
        self.population = self.population[order]

    def objective_func(self, *args):
        self.nfev += 1
        return self.func(*args)
    
    def mutation(self, alpha=0.2, epsilon=0.001):
        new_population = self.population.copy()
        new_sigmas = self.sigmas.copy()
        nvar = len(self.bounds)
        for v in range(nvar):
            new_sigmas[:, v] = np.maximum(epsilon, new_sigmas[:, v] * (1 + self.rng.normal(loc=0, scale=alpha)))
            
            new_population[:, v] += self.rng.normal(loc=0, scale=new_sigmas[:, v])           
            new_population[:, v] = np.maximum(np.ones(self.population.shape[0])*self.bounds[v][0], new_population[:, v])   
            new_population[:, v] = np.minimum(np.ones(self.population.shape[0])*self.bounds[v][1], new_population[:, v])          
        return new_population
                                                     
                                                      
    def survivor_selection(self, new_population):
        two_populations = np.concatenate((self.population, new_population), axis=0)
        two_fitness = self.objective_func(two_populations, *self.args)
        order = np.argsort(two_fitness)
        self.population = two_populations[order[:self.population.shape[0]]]
        self.fitness = two_fitness[order[:self.population.shape[0]]]
        
    
    def solve(self, max_it=1000):
        self.init_population()
        elite_fx = self.fitness[0]
        elite_sol = self.population[0]
        nit = 0
        while nit < max_it and elite_fx > 0:
            new_population = self.mutation() 
            self.survivor_selection(new_population)
            elite_fx = self.fitness[0]
            elite_sol = self.population[0]     
            nit += 1

        ans = {'x': elite_sol, 'nit': nit, 'fun': elite_fx, 'nfev': 0}                                
        # x: solution
        # nit: generation count
        # fun: fitness of the fittest
        # nfev: call count to objective function
        return ans

def evolutionary_programming(func, bounds, args=(), popsize=30):
    print('Evolutionary Programming')
    ep = EvolutionaryProgramming(func, bounds, args, popsize)
    return ep.solve()