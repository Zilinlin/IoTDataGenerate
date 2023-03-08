# this file for the basic algorithm PSO
# the PSO algorithm is for one window
# written by Zilin
import numpy as np
import random
from feature_extractor import FeatureExtractor
from numpy_generator import NumpyGenerator

class PSO:
    def __init__(self, learning_rate, max_generation, window, range_speed, range_pop, algorithm):
        self.window = window
        self.packets = window.get_packets("total")
        self.lr = learning_rate #the accelaration constant
        self.max_gen = max_generation #the generation number
        self.size_pop = len(self.packets) #the size of population
        self.range_speed = range_speed #the limitation of speed range will be like [(0,0.02),(0,2)]

        #each particle is one packet
        self.pop = [] # the population position for each particle,
        self.v = [] # the velocity for each particle
        self.fitness = [] #the fitness score for each particle, but the score will be the same

        self.p_best_pop = []
        self.p_best_fit = []

        self.range_pop = range_pop #the population range will be like [(0,0.4)s,(0,40)bit]
        # the algorithm helps detection
        self.alg = algorithm

        self.fe = FeatureExtractor(None)

        self.cal_fitness()


    #calculate the fitness y of position x
    def cal_fitness(self):
        # TODO finish the fitness calculation
        # calculate the feature
        self.fe.process_window(self.window)
        data_generator = NumpyGenerator([self.window],"reconnaissance")
        data_generator.process_windows()
        data = data_generator.dataset
        label = data_generator.label

        # predicts the result of currrent window
        pred = self.alg.cal_fitness(data,label,"reconnaissance")

        print("the prediction of PSO window:",pred)

    # population is n*2, first is ts+, packet_length+
    #initiate population and fitness value
    def initiate_pop_fit(self):
        self.pop = np.zeros((self.size_pop,2))
        self.v = np.zeros((self.size_pop,2))
        self.fitness = np.zeros((self.size_pop))

        for i in range(self.size_pop):
            self.pop[i] = [random.uniform(self.range_pop[0][0],self.range_pop[0][1]), random.randint(self.range_pop[1][0], self.range_pop[1][1])]
            self.v[i] = [random.uniform(self.range_speed[0][0], self.range_speed[0][1]), random.randint(self.range_speed[1][0], self.range_speed[1][1])]

            # initiate the fitness
            self.cal_fitness()





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



