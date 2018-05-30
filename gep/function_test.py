import genome
import math
import random
import numpy as np


#def sr(arg_list):
#    return math.sqrt(arg_list[0])

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sm(arg_list):
    return arg_list[0] + arg_list[1]


def df(arg_list):
    return arg_list[0] - arg_list[1]


def mt(arg_list):
    return arg_list[0] * arg_list[1]


def sq(arg_list):
    return math.pow(arg_list[0], 2)


f_map = {#'S': {'func': sr, 'n': 1},
         'q': {'func': sq, 'n': 1},
         's': {'func': sm, 'n': 2},
         'd': {'func': df, 'n': 2},
         'm': {'func': mt, 'n': 2}}

n_tests = 50
mutation_rate = 0.05


def test(test_func, len_h, n_in, generations=100, n_pop=50):
    
    A = genome.GEP(f_map, n_in)
    
    genome_length = A.genome_length(len_h)
    
    population = list()
    
    for i in range(n_pop):
        population.append({'g': A.random_genome(len_h), 'f': 0.0})
    
    # Generational loop
    for i in range(generations):
        
        tests = list()
        
        # Build a set of random test cases
        for j in range(n_tests):
            foo = [random.uniform(0.0, 100.0) for _ in range(n_in)]
            tests.append([foo, test_func(foo)])
        
        # Test population
        for j in population:
            
            p = A.phenotype(j['g'])
            cum_err = 0.0
            
            for k in tests:
                cum_err += math.sqrt(math.pow(p.process(k[0])-k[1], 2))
                
            j['f'] = sigmoid(1.0/(cum_err/n_tests)) if cum_err > 0.0 else 1.0
            
        fits = [j['f'] for j in population]
        cum_sum = np.cumsum(fits)
        new_pop = list()

        print("Max F: {:.5f}, Avg F: {:.5f} ".format(max(fits), sum(fits)/n_pop), population[fits.index(max(fits))]['g'])
        
        for j in range(int(n_pop/2)):
            n1 = random.uniform(0.0, cum_sum[-1])
            i1 = 0
            while n1 > cum_sum[i1]:
                i1 += 1

            n2 = random.uniform(0.0, cum_sum[-1])
            i2 = 0
            while n2 > cum_sum[i2]:
                i2 += 1
                
            cross_point = random.randint(0, genome_length)
            
            new1 = population[i1]['g'][:cross_point]+population[i2]['g'][cross_point:]
            new2 = population[i2]['g'][:cross_point]+population[i1]['g'][cross_point:]
            
            for k in range(len_h):
                if random.random() < mutation_rate:
                    new1 = new1[:k]+random.choice(A.head_alleles)+new1[k+1:]
                if random.random() < mutation_rate:
                    new2 = new2[:k]+random.choice(A.head_alleles)+new2[k+1:]
                    
            for k in range(len_h, genome_length):
                if random.random() < mutation_rate:
                    new1 = new1[:k] + random.choice(A.index) + new1[k + 1:]
                if random.random() < mutation_rate:
                    new2 = new2[:k] + random.choice(A.index) + new2[k + 1:]
            
            new_pop.extend([{'g': new1, 'f': 0.0}, {'g': new2, 'f': 0.0}])
        
        population = new_pop
            
            
test(lambda x: math.pow(x[0], 2)-x[0], 4, 1, 100, 20)