import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from student_utils_sp18 import *
import networkx as nx
import numpy as np
from pytsp import atsp_tsp, run, dumps_matrix
from heapq import heappop, heappush

"""
======================================================================
  Complete the following function.
======================================================================
"""

def solver_helper(list_of_kingdom_names, H, starting_status, adjacency_matrix):
    """
    Write your algorithm here.
    Input:
        list_of_kingdom_names: An list of kingdom names such that node i of the graph corresponds to name index i in the list
        starting_kingdom: The name of the starting kingdom for the walk
        adjacency_matrix: The adjacency matrix from the input file

    Output:
        Return 2 things. The first is a list of kingdoms representing the walk, and the second is the set of kingdoms that are conquered
    """

    N = len(list_of_kingdom_names)

    starting_owned = sum([1 for i in starting_status if i > 0])

    if starting_owned == N:
        return starting_status
    if N == 1:
        starting_status[0] = 2
        return starting_status
    starting_status = tuple(starting_status)

    starting_index = 0
    lengths = dict()
    for i in range(N):
        length, path = nx.bidirectional_dijkstra(H, 0, i)
        lengths[(-1, i)] = length
        lengths[(i, -1)] = length
        lengths[(0, i)] = length
        lengths[(i, 0)] = length

    # dfs to construct a graph of graphs
    queue = []
    known = set()


    s = (starting_index, starting_owned, starting_status)
    found = set()
    smallest = float('inf')

    queue.append((0, (-1, starting_owned, starting_status)))
    while queue != []:
        cost, node = heappop(queue)
        if node not in known:
            cur, n, status = node
            known.add(node)

            if cur == starting_index and n == N:
                if cost > smallest:
                    return found
                # found.add(status)
                # smallest = cost
                return status
            if n == N:
                heappush(queue, (cost + lengths[(cur, 0)], (0, n, status)))
            for next in range(N):
                if status[next] < 2:
                    new_status = list(status)
                    new_n = n
                    if new_status[next] < 1:
                        new_n = n + 1
                    new_status[next] = 2
                    for neighbor in H.adj[next]:
                        if new_status[neighbor] < 1:
                            new_status[neighbor] = 1
                            new_n += 1
                    new_node = (next, new_n, tuple(new_status))
                    if (cur, next) not in lengths:
                        length, _ = nx.bidirectional_dijkstra(H, cur, next)
                        lengths[(cur, next)] = length
                        lengths[(next, cur)] = length
                    length = lengths[(cur, next)] + adjacency_matrix[list_of_kingdom_names[next]][list_of_kingdom_names[next]]
                    heappush(queue, (cost + length, new_node))

def solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_kingdom_names: An list of kingdom names such that node i of the graph corresponds to name index i in the list
        starting_kingdom: The name of the starting kingdom for the walk
        adjacency_matrix: The adjacency matrix from the input file

    Output:
        Return 2 things. The first is a list of kingdoms representing the walk, and the second is the set of kingdoms that are conquered
    """

    split_size = 15
    subgraph = nx.Graph()
    last_node = None


    starting_status = [0 for _ in list_of_kingdom_names]
    sublist = []

    starting_index = list_of_kingdom_names.index(starting_kingdom)
    N = len(list_of_kingdom_names)
    H = adjacency_matrix_to_graph(adjacency_matrix)

    result = []
    closed_walk, conquered_kingdoms = [], []

    stack = []
    known = set()
    stack.append((None, starting_index))
    while stack:
        prev, node = stack.pop()
        if node in known:
            continue
        sublist.append(node)
        known.add(node)
        flag = False

        if len(sublist) == 1:
            subgraph.add_node(0)
        elif prev in sublist:
            subgraph.add_edge(sublist.index(prev), len(sublist)-1, weight=H[prev][node]['weight'])
        else:
            flag = True
            sublist.pop()
        
        if flag or len(sublist) == split_size or len(known) == N:
            sub_starting_status = [starting_status[node] for node in sublist]

            sub_result = solver_helper(sublist, subgraph, sub_starting_status, adjacency_matrix)
            for i in range(len(sub_result)):
                if sub_result[i] == 2:
                    conquered_kingdoms.append(sublist[i])
                    for nbr in H.adj[sublist[i]]:
                        starting_status[nbr] = 1
            subgraph = nx.Graph()
            sublist = []
        if flag:
            subgraph.add_node(0)
            sublist.append(node)

        # Need to fix the order the neigbhors being added: too costly
        adjs = list(H.adj[node])
        avg = sum([H[node][nbr]['weight'] for nbr in adjs])/len(adjs)
        adjs = sorted(adjs, key=lambda nbr: abs(H[node][nbr]['weight']-avg), reverse=True)
        for nbr in adjs:
            stack.append((node, nbr))
    if sublist != []:
        sub_starting_status = [starting_status[node] for node in sublist]
        sub_result = solver_helper(sublist, subgraph, sub_starting_status, adjacency_matrix)
        for i in range(len(sub_result)):
            if sub_result[i] == 2:
                conquered_kingdoms.append(sublist[i])



    tmp_conquered = [(-adjacency_matrix[c][c], c) for c in conquered_kingdoms]

    while tmp_conquered != []:
        c = heappop(tmp_conquered)[1]
        conquered_kingdoms.remove(c)
        if not nx.is_dominating_set(H, conquered_kingdoms):
            conquered_kingdoms.append(c)

    conquered = list(conquered_kingdoms)

    if starting_index not in conquered:
        conquered.append(starting_index)

    C = len(conquered)
    if C == 1:
        return [starting_kingdom], [starting_kingdom]
    elif C == 2:
        if starting_index == conquered[0]:
            start, next = conquered
        else:
            next, start = conquered
        _, path = nx.bidirectional_dijkstra(H, start, next)
        path = [list_of_kingdom_names[c] for c in path]
        return path + path[::-1][1:], [list_of_kingdom_names[c] for c in conquered_kingdoms]
    else:
        matrix = np.zeros((C, C))
        paths = dict()
        for i in range(C):
            for j in range(i+1, C):
                matrix[i, j], path = nx.bidirectional_dijkstra(H, conquered[i], conquered[j])
                paths[(i, j)] = path
                paths[(j, i)] = path[::-1]
        matrix = matrix + matrix.T

        if np.max(matrix) >= 10e6:
            matrix = matrix / 10e6
        outf = "/tmp/myroute.tsp"
        with open(outf, 'w') as dest:
            dest.write(dumps_matrix(matrix, name="My Route"))

        tour = run(outf, start=conquered.index(starting_index), solver="LKH")

        tour_path = tour['tour']
        for i in range(len(tour_path)):
            cur = tour_path[i]
            if i == len(tour_path)-1:
                dest = conquered.index(starting_index)
            else:
                dest = tour_path[i+1]
            closed_walk.append(paths[(cur, dest)])

        final_walk = []

        for i in range(len(closed_walk)):
            data = closed_walk[i]

            if i == 0:
                for j in data:
                    final_walk.append(list_of_kingdom_names[j])
            else:
                for j in data[1:]:
                    final_walk.append(list_of_kingdom_names[j])

        return final_walk, [list_of_kingdom_names[c] for c in conquered_kingdoms]

"""
======================================================================
   No need to change any code below this line
======================================================================
"""


def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix = data_parser(input_data)
    closed_walk, conquered_kingdoms = solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    output_filename = utils.input_to_output(filename)
    output_file = '{0}/{1}'.format(output_directory, output_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    utils.write_data_to_file(output_file, closed_walk, ' ')
    utils.write_to_file(output_file, '\n', append=True)
    utils.write_data_to_file(output_file, conquered_kingdoms, ' ', append=True)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
