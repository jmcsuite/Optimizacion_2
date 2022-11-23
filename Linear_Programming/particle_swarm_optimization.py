from cmath import inf
import numpy as np

class ParticleOptimization:
    def __init__(self, func, bounds, args, A_ub, b_ub, A_eq, b_eq, popsize=30, rng=np.random.default_rng(6502729091)):
        self.A_ub = np.array(A_ub)
        self.A_eq = np.array(A_eq)
        self.b_ub = np.array(b_ub)
        self.b_eq = np.array(b_eq)
        self.func = func
        self.bounds = np.array(bounds).astype(np.float64)
        for i,a in enumerate(self.bounds[:,1]):
            if(np.isnan(a)):
                self.bounds[i,1] = 100.0

        for i,a in enumerate(self.bounds[:,0]):
            if(np.isnan(a)):
                self.bounds[i,0] = -100.0
        #print(self.bounds)
        self.args = args
        self.popsize = popsize
        self.population = None
        self.velocity = None
        self.bestPositions = None
        self.bestPosition = None
        self.fitness = None
        self.nfev = 0
        self.rng = rng
        
    #Calcularlo de manera matricia. Recibir muchas soluciones y devover vectores de penalty
    def objective_func(self, *args):
        self.nfev += 1
        x = args[0]
        a1 = self.bounds[:,0]-x
        a2 = x-self.bounds[:,1]
        a1 = np.exp(a1+7)
        
        a1 = np.sum(a1)
        a2 = np.exp(a2+7)
        a2 = np.sum(a2)
        
        a3 = 0
        if(self.A_ub.shape != ()):
            matrixIn = self.A_ub@x
            a3 = matrixIn - self.b_ub
            a3 = np.sum(np.exp(a3+7))
        #print(a1, a2, a3, x)

        a4 = 0
        if(self.A_eq.shape != ()):
            matrixIn = self.A_eq@x
            a4 = np.abs(matrixIn-self.b_eq)
            a4 = np.sum(np.exp(a4+7))
        

        m = self.func(*args)
        return m + a1 + a2 + a3 + a4

    
    def init_population(self):
        nvar = len(self.bounds)
        self.population = np.zeros((self.popsize, nvar))
        
        self.velocity = np.ones((self.popsize, nvar))
        
        for v in range(nvar):
            vmin, vmax = self.bounds[v, 0], self.bounds[v, 1]
            self.population[:,v] = self.rng.uniform(vmin, vmax, (self.popsize))
        self.bestPositions = self.population.copy()
        #self.fitness = self.objective_func()
        self.fitness = np.zeros((self.popsize))
        for i in range(self.popsize):
            self.fitness[i] = self.objective_func(self.population[i], *self.args)
        argMin = np.argmin(self.fitness)
        self.bestPosition = self.population[argMin]
    
    def mutation(self, bestPop, position, velocity, w, c1, c2):
        r1 = self.rng.uniform(0,1)
        r2 = self.rng.uniform(0,1)
        velocity2 = w*velocity + c1*r1*(bestPop-position) + c2*r2*(self.bestPosition-position)
        position2 = position + velocity2
        # check bounds
        return position2, velocity2
        #return np.clip(vi, self.bounds[:,0], self.bounds[:,1])
    
    def crossover(self, xi, v, Cr):
        nvar = len(self.bounds)
        l = self.rng.integers(0, nvar)
        u = np.zeros((nvar))
        for i in range(nvar):
            u[i] = 0 if self.rng.uniform(0, 1) <= Cr else 1
        u[l] = 0.
        # 0 -> from mutated, 1 -> from original
        return xi * u + v * ((u + 1) % 2)
    
    def solve(self, max_it):
        self.init_population()

        elite_sol = self.bestPosition
        elite_fx = self.objective_func(elite_sol, *self.args)
        #print(self.objective_func(elite_sol, *self.args), elite_fx, "pp")
        w, c1, c2 = 0.9, 0.9, 0.9
        current_g = 0
        while current_g < max_it:
            w *= .99
            c1 *= .99
            c2 *= .99
            for idx in range(self.popsize):
                pi , vi= self.mutation(self.bestPositions[idx], self.population[idx], self.velocity[idx], w, c1, c2)
                #ui = self.crossover(self.population[idx], vi, Cr)
                fui = self.objective_func(pi, *self.args)
                self.population[idx] = pi
                self.velocity[idx] = vi
                if fui < self.fitness[idx]:
                    self.fitness[idx] = fui 
                    self.bestPositions[idx] = pi
                if self.fitness[idx] < elite_fx:
                    elite_fx = self.fitness[idx]
                    elite_sol = self.population[idx]
                    self.bestPosition = elite_sol
            current_g += 1
        
        #print(self.objective_func(elite_sol, *self.args), elite_fx)

        return {'x': elite_sol, 'nit': current_g, 'fun': elite_fx, 'nfev': self.nfev}
    
def particle_optimization(func, bounds, args, A_ub, b_ub, A_eq, b_eq, popsize=30, max_it=200):
    po = ParticleOptimization(func, bounds, args, A_ub, b_ub, A_eq, b_eq, popsize)
    return po.solve(max_it)
        
       