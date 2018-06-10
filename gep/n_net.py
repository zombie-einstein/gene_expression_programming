import numpy as np
from genome import GEP
from random import getrandbits, randint, uniform


def crossover_1(ga, gb):
    cp = randint(0, len(ga))
    return ga[:cp] + gb[cp:], gb[:cp] + ga[cp:]


def crossover_2(ga, gb):
    cps = [randint(0, len(ga)), randint(0, len(ga))]
    cps.sort()
    return ga[:cps[0]] + gb[cps[0]:cps[1]] + ga[cps[1]:], gb[:cps[0]] + ga[cps[0]:cps[1]] + gb[cps[1]:]


f_map = {'A': {'func': lambda x, y: x and y, 'n': 2},
         'O': {'func': lambda x, y: x or y, 'n': 2},
         'N': {'func': lambda x: not x, 'n': 1},
         'E': {'func': lambda x, y: x == y, 'n': 2},
         'Q': {'func': lambda x, y: x != y, 'n': 2}}

head_length_in = 15
head_length_up = 15

weight_func = GEP(f_map, 3, head_length_in)
update_func = GEP(f_map, 4, head_length_up)


class NNet:
    def __init__(self, in_genome, up_genome):
        self.in_genome = in_genome
        self.up_genome = up_genome
        self.in_func, _ = weight_func.pre_phenotype(self.in_genome)
        self.up_func, _ = update_func.pre_phenotype(self.up_genome)
        self.weights = [True, True, True]
        self.activation = True
        self.inputs = [False, False, False]

    def reset(self):
        self.weights = [True, True, True]
        self.activation = True
        self.inputs = [False, False, False]

    def new_input(self, in_arr):
        self.inputs = [a and b for a, b in zip(in_arr, self.weights)]
    
    def run(self):
        self.activation = self.in_func(self.inputs)
        
    def update_weights(self, target):
        self.weights = [self.up_func([target, self.activation, self.weights[i], self.inputs[i]]) for i in range(3)]
        
    def train_step(self, in_arr, target):
        self.new_input(in_arr)
        self.run()
        self.update_weights(target)
        return self.activation
    
    def predict_step(self, in_arr):
        self.new_input(in_arr)
        self.run()
        return self.activation


POP = 50
GENERATIONS = 400
TESTS = 200
MUTATION_RATE = 0.001

test_funcs = [lambda x: x[0] and x[1],
              lambda x: x[0] and x[2],
              lambda x: x[1] and x[2],
              lambda x: x[0] or x[1],
              lambda x: x[0] or x[2],
              lambda x: x[1] or x[2]]

population = [NNet(weight_func.random_genome(), update_func.random_genome()) for _ in range(POP)]

for p in population:
    print(p.in_genome, p.up_genome)

for _ in range(GENERATIONS):
    
    tests = [[bool(getrandbits(1)), bool(getrandbits(1)), bool(getrandbits(1))] for j in range(TESTS)]
    targets = [[y(x) for x in tests] for y in test_funcs]
    
    ind_score = [[0 for _ in range(len(test_funcs))] for _ in range(len(population))]
    
    for j, p in enumerate(population):
        for t in range(len(targets)):
            for k in range(int(7*TESTS/8)):
                p.train_step(tests[k], targets[t][k])
            for k in range(int(7 * TESTS / 8), TESTS):
                if p.predict_step(tests[k]) == targets[t][k]:
                    ind_score[j][t] += 1
            p.reset()
    
    score = [int(np.max(i)/np.std(i)) for i in ind_score]
    cum_sum = np.cumsum(score)
    new_pop = list()

    top2 = sorted(range(len(score)), key=lambda i: score[i])[-2:]
    
    for i in top2:
        new_pop.append(population[i])
    
    for j in range(int((POP-2) / 2)):
        n1 = uniform(0.0, cum_sum[-1])
        i1 = 0
        while n1 > cum_sum[i1]:
            i1 += 1
    
        n2 = uniform(0.0, cum_sum[-1])
        i2 = 0
        while n2 > cum_sum[i2]:
            i2 += 1
        
        new_in_1, new_in_2 = crossover_1(population[i1].in_genome, population[i2].in_genome)
        new_up_1, new_up_2 = crossover_1(population[i1].up_genome, population[i2].up_genome)
        
        new_in_1 = weight_func.mutate_genome(MUTATION_RATE, new_in_1)
        new_in_2 = weight_func.mutate_genome(MUTATION_RATE, new_in_2)

        new_up_1 = update_func.mutate_genome(MUTATION_RATE, new_up_1)
        new_up_2 = update_func.mutate_genome(MUTATION_RATE, new_up_2)
    
        new_pop.append(NNet(new_in_1, new_up_1))
        new_pop.append(NNet(new_in_2, new_up_2))

    population = new_pop
    print(len(set([x.in_genome for x in population])),
          len(set([x.up_genome for x in population])),
          np.mean(score))

for i, p in enumerate(population):
    print(p.in_genome, p.up_genome, ind_score[i], score[i])
