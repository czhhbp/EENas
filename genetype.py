import random
import sys
import copy
sys.path.append('..')
from google_space_model import PRIMITIVES
op_num = len(PRIMITIVES)

"""
genetype:
一个个体表示为两个字典，分别代表normal cell和reduction cell
两个实数字典表示形式一致，以normal cell为例：
[
[[in_node_id1, edge_type], [in_node_id2, edge_type]],
[[in_node_id1, edge_type], [in_node_id2, edge_type]],
[[in_node_id1, edge_type], [in_node_id2, edge_type]],
[[in_node_id1, edge_type], [in_node_id2, edge_type]]
]
{
node_id0: [[in_node_id1, edge_type], [in_node_id2, edge_type]],
node_id1: [[in_node_id1, edge_type], [in_node_id2, edge_type]],
node_id2: [[in_node_id1, edge_type], [in_node_id2, edge_type]],
node_id3: [[in_node_id1, edge_type], [in_node_id2, edge_type]],
}

"""



class POP:

    def __init__(self, population=20, node_ids=[2, 3, 4, 5], in_node_num=2, op_num=7,pm=0.1):
        self.node_ids = node_ids
        self.in_node_num = in_node_num
        self.op_num = op_num
        self.node_num = len(node_ids)
        self.pm = pm
        self.population = population
        self.children = []
        self.win_count = [1] * population
        for i in range(self.population):
            child = self.random_identity()
            self.children.append(child)

    def random_identity(self):
        identity = []
        for i in range(2):
            cell = []
            for node_id in self.node_ids:
                node_gene1 = [random.randint(0, node_id - 1), random.randint(0, self.op_num - 1)]
                node_gene2 = [random.randint(0, node_id - 1), random.randint(0, self.op_num - 1)]
                cell.append([node_gene1, node_gene2])
            identity.append(cell)
        return identity

    def mutate(self, child, combine=False):
        new_child = copy.deepcopy(child)
        # mutate for in_node id or op
        cell_id = random.randint(0, 1)
        cell = new_child[cell_id]
        node_id = random.randint(self.node_ids[0], self.node_ids[-1])
        node = cell[node_id - 2]
        element_id = random.randint(0, 1)
        element = node[element_id]

        if random.random() < 0.5:
            element[0] = (element[0] + random.randint(1, node_id - 1)) % node_id
        else:
            element[1] = (element[1] + random.randint(1, self.op_num - 1)) % self.op_num
        if combine:
            combine_child = copy.deepcopy(child)
            combine_child[cell_id][node_id].extend(new_child[cell_id][node_id])
            return new_child, combine_child
        return new_child

    def large_mutate(self, child):
        new_child = copy.deepcopy(child)
        for cell_id in range(2):
            for nodeORop in range(2):
                node_id = random.randint(0, self.node_num - 1)
                element_id = random.randint(0, 1)
                choice_num = node_id + 2 if nodeORop == 0 else self.op_num
                new_child[cell_id][node_id][element_id][nodeORop] = (new_child[cell_id][node_id][element_id][nodeORop]
                                                                     + random.randint(1, choice_num - 1)) % choice_num
        return new_child

    def update(self, unselected=0, selected=0, large_mutate=False, combine=False):
        if large_mutate:
            new_child = self.large_mutate(self.children[selected])
        else:
            new_child = self.mutate(self.children[selected], combine)
        self.children[unselected] = new_child
        self.win_count[selected] += 1
        self.win_count[unselected] = 0

    def get_two_gene(self):
        gene_ids = random.sample(list(range(self.population)), 2)
        child1, child2 = self.children[gene_ids[0]], self.children[gene_ids[1]]
        return gene_ids, [child1, child2]

    def get_gene(self, child_id):
        return self.children[child_id]

    def get_winners(self):
        winners = [self.children[i] for i in range(self.population) if self.win_count[i] > 0]
        return winners

    def pop_num(self):
        return self.population

















