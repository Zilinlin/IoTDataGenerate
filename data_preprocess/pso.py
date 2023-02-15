# this file for the basic algorithm PSO
# written by Zilin

class PSO:
    def __init__(self,learning_rate, max_generation, size_pop, range_speed):
        self.lr = learning_rate #the accelaration constant
        self.max_gen = max_generation #the generation number
        self.size_pop = size_pop #the size of population
        self.range_speed = range_speed #the limitation of speed range

        #TODO: each particle should be one packet?
        self.pop = [] # the population position for each particle
        self.v = [] # the velocity for each particle
        self.fitness = [] #the fitness score for each particle

        self.p_best_pop = []
        self.p_best_fit = []

    #calculate the fitness y of position x
    def cal_fitness(self,x):
        # TODO finish the fitness calculation
        y = 0
        return y

    # TODO how to define population (packets? vector)
    #initiate population and fitness value
    def initiate_pop_fit(self):


    # calculate the inital global and particle best value
    # fitness and pop is initial
    def get_intial_best(self, fitness, pop):
        # the best particle with max fitness
        g_best_pop, g_best_fit = pop[fitness.argmax()].copy(),fitness.max()
        p_best_pop, p_best_fit = pop.copy(), fitness.copy()
        return g_best_pop, g_best_fit, p_best_pop, p_best_fit

    def run_pso(self):
        self.initiate_pop_fit()

        max_gen = self.max_gen
        size_pop = self.size_pop
        lr = self.lr

        for i in range(max_gen):
            t = 0.5

            #update velocity for each particle
            for j in range(size_pop):
                v[j] += lr[0]*np.random.rand()*(self.p_best_pop[j]-pop[j]) + lr[1]*np.random.rand()*(self.g_best_pop - pop[j])
            v[v<self.range_speed[0]] = self.range_speed[0]
            v[v>self.range_speed[1]] = self.range_speed[1]

            #update the fitness
            for j in range(size_pop):
                self.fitness[j] = self.cal_fitness(self.pop[j])

            #update the global/particle_best_population/fitness
            for j in range(size_pop):
                if self.fitness[j] > self.p_best_fit[j]:
                    self.p_best_fit[j] = self.fitness[j]
                    self.p_best_pop[j] = self.pop[j].copy()

            if self.p_best_fit.max() > self.g_best_fit:
                self.g_best_fit =  self.p_best_fit.max()
                self.g_best_pop = self.pop[self.p_best_fit.argmax()].copy()


    def get_learning_rate(self):
        return self.lr

    def get_max_generation(self):
        return self.max_gen

    def get_size_population(self):
        return self.size_pop

    def get_range_speed(self):
        return self.range_speed



