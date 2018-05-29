from collections import deque
from random import choice


class LeafNode:
    """Leaf node, just returns value from an argument array, obviously has no children nodes"""
    def __init__(self, n):
        self.func = lambda arr: arr[n]
        self.n = 0
    
    def process(self, arr):
        return self.func(arr)


class FuncNode:
    """Intermediate function leaf nodes, return their function with child return values passed as list of arguments"""
    def __init__(self, func, n):
        self.func = func
        self.n = n
        self.child = []
    
    def process(self, arr):
        """Return the value of this node from results of children nodes. A func node may be the
        root of the function tree and so this function returns the final result of the tree function"""
        return self.func([i.process(arr) for i in self.child])


class GEP:
    """GEP utility class initialized on a allele <-> function mapping dictionary and input size
    (which will be passed as list to the generated function). This class both generates random genotypes based on the
    function/input mappings and converts said genotypes into executable function graphs"""

    def __init__(self, func_map, n_args):
        self.func_map = func_map
        self.n_args = n_args
        self.max_args = max([i['n'] for i in self.func_map.values()])

    def make_node(self, c):
        """Return an appropriate node based on an input character"""
        if c.isdigit():
            return LeafNode(int(c))
        else:
            f = self.func_map[c]
            return FuncNode(f['func'], f['n'])

    def phenotype(self, genome):
        """From a genotype string return a executable function graph"""
        root = self.make_node(genome[0])
        a = deque([root])
        b = deque(deque(genome[1:]))
    
        while a:
            curr = a.popleft()
            for i in range(curr.n):
                new_node = self.make_node(b.popleft())
                curr.child.append(new_node)
                a.append(new_node)
    
        return root

    def random_genome(self, len_h):
        """Generate a random genome fitting the class parameters with argument genome head length"""
        len_t = len_h * (self.max_args - 1) + 1
        index = [str(i) for i in range(self.n_args)]
        head_alleles = list(self.func_map.keys())+index
        ret = ''
        for i in range(len_h):
            ret += choice(head_alleles)
        for i in range(len_t):
            ret += choice(index)
        return ret
