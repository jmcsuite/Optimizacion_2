import numpy as np

class DifferentialEvolution:
    def __init__(self, func, bounds, args=(), popsize=30, rng=np.random.default_rng(6502729091)):
        self.func = func
        self.bounds = np.array(bounds)
        self.args = args
        self.popsize = popsize
        self.population = None
        self.fitness = None
        self.nfev = 0
        self.rng = rng
        
    def objective_func(self, *args):
        self.nfev += 1
        return self.func(*args)
    
    def init_population(self):
        nvar = len(self.bounds)
        self.population = np.zeros((self.popsize, nvar))
        
        for v in range(nvar):
            vmin, vmax = self.bounds[v, 0], self.bounds[v, 1]
            self.population[:,v] = self.rng.uniform(vmin, vmax, (self.popsize))
    
    def mutation(self):
        r = self.rng.integers(0, self.popsize, size=3)
        while r[1] == r[0]:
            r[1] = self.rng.integers(0, self.popsize)
        while r[2] == r[1] or r[2] == r[0]:
            r[2] = self.rng.integers(0, self.popsize)
        vi = self.population[r[0]] + self.rng.uniform(0, 2) * (self.population[r[1]] - self.population[r[2]])
        # check bounds
        return np.clip(vi, self.bounds[:,0], self.bounds[:,1])
    
    def crossover(self, xi, v, Cr):
        nvar = len(self.bounds)
        l = self.rng.integers(0, nvar)
        u = np.zeros((nvar))
        for i in range(nvar):
            u[i] = 0 if self.rng.uniform(0, 1) <= Cr else 1
        u[l] = 0.
        # 0 -> from mutated, 1 -> from original
        return xi * u + v * ((u + 1) % 2)
    
    def solve(self, max_it, Cr):
        self.init_population()
        self.fitness = np.zeros((self.popsize))
        
        for i in range(self.popsize):
            self.fitness[i] = self.objective_func(self.population[i], *self.args)

        elite_fx = self.fitness[0]
        elite_sol = self.population[0]
        current_g = 0
        while current_g < max_it and elite_fx > 0:
            for idx in range(self.popsize):
                vi = self.mutation()
                ui = self.crossover(self.population[idx], vi, Cr)
                fui = self.objective_func(ui, *self.args)
                if fui < self.fitness[idx]:
                    self.fitness[idx] = fui 
                    self.population[idx] = ui
                if self.fitness[idx] < elite_fx:
                    elite_fx = self.fitness[idx]
                    elite_sol = self.population[idx]
            current_g += 1
        
        return {'x': elite_sol, 'nit': current_g, 'fun': elite_fx, 'nfev': self.nfev}
    
def differential_evolution(func, bounds, args=(), popsize=30, max_it=3000, Cr=0.5):
    de = DifferentialEvolution(func, bounds, args, popsize)
    return de.solve(max_it, Cr)
        
       