    
class Graph:
    class Vertex:
        """Vertex structure for a graph"""

        __slots__ = ['_element']

        def __init__(self, x):
            """Do not call the constructor directly, use Graph's insert_vertex instead"""
            self._element = x

        def element(self):
            """Returns the element associated with this vertex"""
            return self._element

        def __hash__(self):
            return hash(id(self))

    class Edge:
        """Edge structure for a graph"""

        __slots__ = ['_origin', '_destination', '_element']
        
        def __init__(self, u, v, x):
            self._origin = u
            self._destination = v
            self._element = x
        
        def element(self):
            """returns the element associated with the edge"""
            return self._element
        
        def endpoints(self):
            """returns the a tuple with following structure (origin, destination)"""
            return (self._origin, self._destination)
        
        def opposite(self, v):
            """assuming that v is one of the edge points will return the other end point"""
            return self._destination if v is self._origin else self._origin

        # This Edge structure will be used for a simple graph. Hence, (org, dest) will be unique
        def __hash__(self):
            return hash((self._origin, self._destination))
    

    def __init__(self, directed=False):
        """
        Create an empty graph 
        By default a undirected graph
        A directed graph if directed is True    
        """
        self._outgoing = {}

        # if undirected set it to self._outgoing
        self._incoming = {} if directed else self._outgoing

    def is_directed(self):
        """Returns true if it is a directed graph"""
        return self._outgoing is not self._incoming
    
    def vertex_count(self):
        """Returns the total number of verticies in the graph"""
        return len(self._outgoing)
    
    def verticies(self):
        """Returns an iterable object"""
        return self._outgoing.keys()
    
    # version 1.1
    # def edge_count(self):
    #     total_edges = sum(len(indicie_map) for indicie_map in self._outgoing.values())
    #     return total_edges if self.is_directed() else total_edges//2
    
    # version 1.2
    def edge_count(self):
        """Returns the total number of edges"""
        total_edges = sum(len(self._outgoing[v]) for v in self._outgoing)

        # undirected edges are represented as (u, v) and (v, u)
        return total_edges if self.is_directed() else total_edges//2

    def edges(self):
        """Returns the set of all the edges"""
        return {edge for v in self._outgoing for edge in self._outgoing[v].values()}
            
    def get_edge(self, u, v):
        """Returns the edge that connects u to v"""
        
        # returns None if v is not accessible from u
        return self._outgoing[u].get(v)
    
    def degree(self, v, outgoing=True):
        """By default returns the number of outgoing edges incident to v.
            
            Optional parameter could be used to get the number of icoming edges incident to v"""
        
        return len(self._outgoing[v]) if outgoing else len(self._incoming[v])
    
    def incident_edges(self, v, outgoing=True):
        """By default returns the iteration of outgoing edges that are incident to v

            Optional parameter can used to get the iteration of incoming edges
        """
        adj_map = self._outgoing[v] if outgoing else self._incoming[v]
        for edge in adj_map.values():
            yield edge

    def insert_vertex(self, x=None):
        """Insert and retrun a new Vertex that holds the value x"""
        v = self.Vertex(x)
        self._outgoing[v] = {}
        if (self.is_directed()):
            self._incoming[v] = {}
        return v
    
    def insert_edge(self, u, v, x=None):
        """Inserts and returns a new edge"""
        e = self.Edge(u, v, x)
        self._outgoing[u][v] = e
        self._incoming[v][u] = e
        return e

    def remove_vertex(self, v):
        """Removes vertex v and all the incidents of v"""

        # Only implementing for the directed case

        # Remove all the incoming edges
        for u in self._incoming.get(v):
             self._outgoing[u].pop(v)
        
        self._incoming.pop(v)

        # Remove all the outgoing edges
        for u in self._outgoing.get(v):
            self._incoming[u].pop(v)

        self._outgoing.pop(v)

