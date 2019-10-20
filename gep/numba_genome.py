from collections import deque
from random import choice, random
from numba import jit


def list_return(n, t):
    """
    Return a function that returns the value from an array
    Args:
        n: Index to return
        t: Argument (numba) type

    Returns:
        function
    """
    @jit(t(t), nopython=True, nogil=True)
    def foo(x):
        return x[n:n+1]
    return foo


def get_func_node_1(f, t, a):
    """
    Return a function that takes one argument input
    Args:
        f: Numpy function
        t: Argument type
        a: Input function

    Returns:
        function
    """
    @jit(t(t), nopython=True, nogil=True)
    def foo(x):
        return f(a(x))
    return foo


def get_func_node_2(f, t, a, b):
    """
        Return a function that takes 2 argument input
        Args:
            f: Numpy function
            t: Argument type
            a: Input function
            b: input function

        Returns:
            function
        """
    @jit(t(t), nopython=True, nogil=True)
    def foo(x):
        return f(a(x), b(x))
    return foo


class LeafNode:
    """Leaf node, just returns value from an argument array, obviously has no children nodes"""
    def __init__(self, f):
        self.n = 0
        self.f = f

    def process(self):
        return self.f


class FuncNode1:
    """Intermediate function leaf nodes, return their function with child return values passed as list of arguments"""
    def __init__(self, f, t):
        self.n = 1
        self.f = f
        self.t = t
        self.child = None
    
    def process(self):
        """Return the value of this node from results of children nodes. A func node may be the
        root of the function tree and so this function returns the final result of the tree function"""
        return get_func_node_1(self.f, self.t, self.child.process())


class FuncNode2:
    """Intermediate function leaf nodes, return their function with child return values passed as list of arguments"""

    def __init__(self, f, t):
        self.n = 2
        self.f = f
        self.t = t
        self.child_a = None
        self.child_b = None

    def process(self):
        """Return the value of this node from results of children nodes. A func node may be the
        root of the function tree and so this function returns the final result of the tree function"""
        return get_func_node_2(self.f, self.t, self.child_a.process(), self.child_b.process())


class GEP:
    """GEP utility class initialized on a allele <-> function mapping dictionary and input size
    (which will be passed as list to the generated function). This class both generates random genotypes based on the
    function/input mappings and converts said genotypes into executable function graphs"""

    def __init__(self, func_map, n_args, len_h, t):
        self.func_map = func_map
        self.n_args = n_args
        self.t = t

        for i in range(self.n_args):
            self.func_map[str(i)] = {'f': list_return(i, t), 'n': 0}

        self.len_h = len_h
        self.max_args = max([i['n'] for i in self.func_map.values()])
        self.len_t = len_h * (self.max_args - 1) + 1
        self.len_g = self.len_h + self.len_t
        self.index = [str(i) for i in range(self.n_args)]
        self.head_alleles = list(self.func_map.keys())+self.index

    def make_node(self, c):
        """Return an appropriate node based on an input character"""
        if c.isdigit():
            return LeafNode(self.func_map[c]['f'])
        else:
            f = self.func_map[c]
            if f['n'] == 1:
                return FuncNode1(f['f'], self.t)
            else:
                return FuncNode2(f['f'], self.t)
    
    # def pre_phenotype(self, genome):
    #     """From a genotype string return a executable function composition and the number of nodes used.
    #     This uses prefix-gene expression, so we visit nodes in DFS pre-order"""
    #     root = self.make_node(genome[0])
    #     a = [root]
    #     b = deque(genome[1:])
    #     count = 1
    #     while a:
    #         if a[-1].n > 0:
    #             a[-1].n -= 1
    #             count += 1
    #             new_node = self.make_node(b.popleft())
    #             a[-1].child.append(new_node)
    #             a.append(new_node)
    #         else:
    #             a.pop()
    #
    #     return root.process(), count
    
    # def phenotype(self, genome):
    #     """From a genotype string return a executable function composition and the number of nodes used"""
    #     root = self.func_map(genome[0])
    #     a = deque([root])
    #     b = deque(genome[1:])
    #     count = 1
    #     while a:
    #         count += 1
    #         curr = a.popleft()
    #         if curr.n == 1:
    #             new_node = self.make_node(b.popleft())
    #             curr.child = new_node
    #             a.append(new_node)
    #         elif curr.n == 2:
    #             new_node_a = self.make_node(b.popleft())
    #             new_node_b = self.make_node(b.popleft())
    #             curr.child_a = new_node_a
    #             curr.child_b = new_node_b
    #             a.append(new_node_a)
    #             a.append(new_node_b)
    #
    #     return root.process(), count

    def phenotype(self, genome):
        """From a genotype string return a executable function composition and the number of nodes used"""
        root = self.make_node(genome[0])
        a = deque([root])
        b = deque(genome[1:])
        count = 1
        while a:
            count += 1
            curr = a.popleft()
            if curr.n == 1:
                new_node = self.make_node(b.popleft())
                curr.child = new_node
                a.append(new_node)
            elif curr.n == 2:
                new_node_a = self.make_node(b.popleft())
                new_node_b = self.make_node(b.popleft())
                curr.child_a = new_node_a
                curr.child_b = new_node_b
                a.append(new_node_a)
                a.append(new_node_b)

        return root.process(), count

    def random_genome(self):
        """Generate a random genome fitting the class parameters with argument genome head length"""
        ret = ''
        for i in range(self.len_h):
            ret += choice(self.head_alleles)
        for i in range(self.len_t):
            ret += choice(self.index)
        return ret
    
    def mutate_genome(self, mutation_rate, genome):
        ret = ""
        for i in range(self.len_h):
            ret += genome[i] if random() > mutation_rate else choice(self.head_alleles)
        for i in range(self.len_h, self.len_h+self.len_t):
            ret += genome[i] if random() > mutation_rate else choice(self.index)
        return ret
