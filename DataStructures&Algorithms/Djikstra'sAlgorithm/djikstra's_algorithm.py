from pqdict import pqdict

def get_weight(G, from_vertex, to_vertex):
    if from_vertex in G and to_vertex in G[from_vertex]:
        return G[from_vertex][to_vertex]
    return None

def initialize_single_source(G, s):
    if s not in G:
        raise ValueError(f"Source vertex {s} does not exist in the graph.")

    predecessor, d = {}, {}
    for vertex in G:
        predecessor[vertex] = None
        d[vertex] = float('inf')

    d[s] = 0
    return d, predecessor

def relax(u, v, w, p, d, G, Q):
    weigth = w(G, u, v)
    if weigth is not None:
        new_d = d[u] + weigth
        if new_d < d[v]:
            d[v] = new_d
            Q[v] = new_d
            p[v] = u

def dijkstra(G, s, w=get_weight):
    d, predecessor = initialize_single_source(G, s)
    Q = pqdict({vertex: d[vertex] for vertex in G})
    while Q:
        u, _ = Q.popitem()
        for vertex in G[u]:
            relax(u, vertex, w, predecessor, d, G, Q)

    return d, predecessor

# set S is used for theorotical reasons
# def dijkstra(G, s, w=get_weight):
#     d, predecessor = initialize_single_source(G, s)
#     S = set()
#     Q = pqdict({vertex: d[vertex] for vertex in G})
#     while Q:
#         u, _ = Q.popitem()
#         S.add(u)
#         for vertex in G[u]:
#             relax(u, vertex, w, predecessor, d, G, Q)

#     return d, predecessor

G1 = {
    's': {'y': 5, 't': 10},
    'y': {'z': 2, 'x': 9, 't': 3},
    't': {'y': 2, 'x': 1},
    'z': {'x': 6, 's': 7},
    'x': {'z': 4}
}

d, p = dijkstra(G1, 's')

print(f"Distances: {d}, predecessors: {p}")

G2 = {
    's': {'y': 5, 't': 10},
    'y': {'x': 9, 't': 3},
    't': {'y': 2, 'x': 1},
    'z': {'x': 6, 's': 7},
    'x': {'z': 4}
}

d, p = dijkstra(G2, 's')

print(f"Distances: {d}, predecessors: {p}")

G3 = {
    's': {'y': 5, 't': 10},
    'y': {'x': 9, 't': 3},
    't': {'y': 2, 'x': 1},
    'z': {'x': 6, 's': 7},
    'x': {}
}

d, p = dijkstra(G3, 's')

print(f"Distances: {d}, predecessors: {p}")

d, p = dijkstra(G3, 'x')

print(f"Distances: {d}, predecessors: {p}")

d, p = dijkstra(G3, 'k')

print(f"Distances: {d}, predecessors: {p}")
