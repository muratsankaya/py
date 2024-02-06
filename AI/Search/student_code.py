from expand import expand
from typing import List
from copy import deepcopy
from queue import Queue

def get_best_node(open, best_node):
	best_score = open[best_node]['distance_from_start'] + open[best_node]['heuristic_distance']

	for node, values in open.items():
		score = values['distance_from_start'] + values['heuristic_distance']

		if score < best_score or (score == best_score and values['heuristic_distance'] < open[best_node]['heuristic_distance'] ):
			best_node = node
			best_score = score

	return best_node

def a_star_search(dis_map, time_map, start, end):
    open, closed = {}, set()
    parents = {start: None}
    open[start] = {"distance_from_start": 0, "heuristic_distance": dis_map[start][end]}
	
    while open:
        node = get_best_node(open, list(open.keys())[0])
        if dis_map[node][end] == 0:
            return construct_path(parents, end) 
		
        children: List[str] = list(filter(lambda child: child not in closed, expand(node, time_map)))
        for child in children:
            new_distance = open[node]['distance_from_start'] + time_map[node][child]
            if child not in open or new_distance < open[child]['distance_from_start']:
                parents[child] = node  
                open[child] = {"distance_from_start": new_distance, "heuristic_distance": dis_map[child][end]}

        closed.add(node)
        del open[node]

    return None  

def construct_path(parents, end):
	path = []
	while end is not None:
		path.append(end)
		end = parents[end]
	return path[::-1]  # Return reversed path

def depth_first_search(time_map, start, end):
	s = [[start]]
	visited = set()
	visited.add(start)
	while(len(s) != 0):
		path = s.pop()
		if(path[-1] == end):
			return path
		children = expand(path[-1], time_map)
		for child in children:
			if child not in visited:
				visited.add(child)
				dc_path = deepcopy(path)
				dc_path.append(child)
				s.append(dc_path)
	return None

def breadth_first_search(time_map, start, end):
	q = Queue()
	q.put([start])
	visited = set()
	visited.add(start)
	while(not q.empty()):
		path = q.get()
		if(path[-1] == end):
			return path
		children = expand(path[-1], time_map)
		for child in children:
			if child not in visited:
				visited.add(child)
				dc_path = deepcopy(path)
				dc_path.append(child)
				q.put(dc_path)
	return None
