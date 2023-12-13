
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
    

    