import genome
import math
import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


f_map = {#'E': {'func': lambda x: math.exp(x), 'n': 1},
         'Q': {'func': lambda x: math.pow(x, 2), 'n': 1},
         'S': {'func': lambda x, y: x+y, 'n': 2},
         'M': {'func': lambda x, y: x-y, 'n': 2},
         'T': {'func': lambda x, y: x*y, 'n': 2},
         'N': {'func': lambda x: -x, 'n': 1}}

n_tests = 50
mutation_rate = 0.05


def test(test_func, len_h, n_in, generations=100, n_pop=50):
    
    A = genome.GEP(f_map, n_in, len_h)
    
    population = list()
    
    for i in range(n_pop):
        population.append({'g': A.random_genome(), 'f': 0.0})
    
    # Generational loop
    for i in range(generations):
        
        tests = list()
        
        # Build a set of random test cases
        for j in range(n_tests):
            foo = [random.uniform(0.0, 100.0) for _ in range(n_in)]
            tests.append([foo, test_func(foo)])
        
        # Test population
        for j in population:
            
            p, c = A.pre_phenotype(j['g'])
            cum_err = 0.0
            
            for k in tests:
                cum_err += math.pow(p(k[0])-k[1], 2)
                
            cum_err = math.sqrt(cum_err/n_tests)
            j['f'] = sigmoid(1.0/(c*cum_err)) if cum_err > 0.0 else 1.0
            
        fits = [j['f'] for j in population]
        cum_sum = np.cumsum(fits)
        new_pop = list()

        print("Gen{:4} => Max F: {:.5f} | Avg F: {:.5f} | Prty: {:.2f} |".format(i, max(fits), sum(fits)/n_pop, 1.0-float(len(set([x['g'] for x in population])))/len(population)), population[fits.index(max(fits))]['g'])
        
        for j in range(int(n_pop/2)):
            n1 = random.uniform(0.0, cum_sum[-1])
            i1 = 0
            while n1 > cum_sum[i1]:
                i1 += 1

            n2 = random.uniform(0.0, cum_sum[-1])
            i2 = 0
            while n2 > cum_sum[i2]:
                i2 += 1
                
            cross_point = random.randint(0, A.len_h+A.len_t)
            
            new1 = population[i1]['g'][:cross_point]+population[i2]['g'][cross_point:]
            new2 = population[i2]['g'][:cross_point]+population[i1]['g'][cross_point:]
            
            new1 = A.mutate_genome(mutation_rate, new1)
            new2 = A.mutate_genome(mutation_rate, new2)
            
            new_pop.extend([{'g': new1, 'f': 0.0}, {'g': new2, 'f': 0.0}])
        
        population = new_pop
            
            
test(lambda x: math.pow(x[0], 2)-x[1], 10, 2, 1000, 20)
