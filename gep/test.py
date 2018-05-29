import genome
import math
import numpy as np


def sr(arg_list):
    return math.sqrt(arg_list[0])


def sm(arg_list):
    return arg_list[0] + arg_list[1]


def df(arg_list):
    return arg_list[0] - arg_list[1]


def mt(arg_list):
    return arg_list[0] * arg_list[1]


def sq(arg_list):
    return math.pow(arg_list[0], 2)


f_map = {'S': {'func': sr, 'n': 1},
         'q': {'func': sq, 'n': 1},
         's': {'func': sm, 'n': 2},
         'd': {'func': df, 'n': 2},
         'm': {'func': mt, 'n': 2}}

A = genome.GEP(f_map, 4)


def test():
    def foo(x):
        return math.sqrt(math.pow((x[0] + x[1]) * (x[2] + x[3]), 2))
    bar = A.phenotype("Sqmss0123")
    for n in range(1000):
        test_case = [np.random.randint(0, 100) for i in range(4)]
        if bar.process(test_case) != foo(test_case):
            print(n, "\tFailed: ", test_case)


test()

for y in range(50):
    B = A.random_genome(6)
    print(B)
    A.phenotype(B)