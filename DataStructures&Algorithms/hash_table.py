
""" 
    This is a simple Hash Table implementation in Python, using seperate chaining for collision handling
    used a c type array for the table, and an ArrayList for seperate chaining 
"""

from ArrayList import ArrayList
import random
import ctypes


class Item:
    def __init__(self, key, value):
        self.key = key 
        self.value = value
        

class HashTable:
    def __init__(self):
        self.size = 0
        self.capacity = 2
        self.table = self.makeArray(self.capacity)
        self.p = 109345121 #got this from a textbook
        self.a = random.randrange(1, self.p)
        self.b = random.randrange(0, self.p)

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        i = self.hashAndCompress(key)
        if(self.table[i] is None):
            raise KeyError("{} is not in the Hashtable".format(key))
        
        for j in range(len(self.table[i])):
            if(self.table[i][j].key == key):
                return self.table[i][j].value

        raise KeyError("{} is not in the Hashtable".format(key))


    def __setitem__(self, key, value):
        i = self.hashAndCompress(key)
        if(self.table[i] is None):
            self.table[i] = ArrayList()
            self.table[i].append(Item(key, value))
            self.size += 1 
            if(self.size >= self.capacity//2): #keep the load factor < 0.5
                self.resize("double")
            return
        
        for j in range(len(self.table[i])):
            if(self.table[i][j].key == key):
                self.table[i][j].value = value
                return
        
        self.table[i].append(Item(key, value))
        self.size += 1
        if(self.size >= self.capacity//2): #keep the load factor < 0.5
                self.resize("double")
        

    def __delitem__(self, key):
        i = self.hashAndCompress(key)
        if(self.table[i] is None):
            raise KeyError("{} is not in the Hashtable".format(key))
        
        for j in range(len(self.table[i])):
            if(self.table[i][j].key == key):
                self.table[i].pop(j)
                self.size -= 1
                if(self.size <= self.capacity//8): #don't let load factor be < 0.125
                    self.resize("shrink") 
                return
            
        raise KeyError("{} is not in the Hashtable".format(key))
            
    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __next__(self):
        
        if(self.i == self.capacity):
            raise StopIteration
        
        while(self.table[self.i] is None or len(self.table[self.i]) == 0):
            self.i += 1
            if(self.i == self.capacity):
                raise StopIteration
            
        #print("table index: {}, lenght of the ArrayList corresponding to the index: {}".format(self.i, len(self.table[self.i])))

        key = self.table[self.i][self.j].key
        self.j += 1
        if(self.j == len(self.table[self.i])):
            self.i += 1
            self.j = 0

        return key

        

    def __iter__(self):
        self.i, self.j = 0, 0
        return self


    """
    [(ai + b) mod p] mod N where a,i,b,p,N are intagers
    st 0 < a < p, 0 <= b < p, N=self.capacity, p is prime
    """
    def hashAndCompress(self, key):
        return  ( (self.a * hash(key) + self.b) % self.p ) % self.capacity

    def resize(self, operation): 
        prevTable, prevCap = self.table, self.capacity

        #capacity must be updated before transferring myDict to the new table !!!
        #its necessary for hashing, see self.hanshAndCompress
        if(operation == "double"):
            self.capacity *= 2
        else:
            self.capacity //= 2
        
        self.table = self.makeArray(self.capacity)
        self.size = 0 #reset the size

        for i in range(prevCap):
            if(prevTable[i] is not None):
                for j in range(len(prevTable[i])):
                    self[prevTable[i][j].key] = prevTable[i][j].value #have to reposition each key value pair

        #print("new size:{}, new capacity: {}".format(self.size, self.capacity))

             
    def makeArray(self, n):
        arr = (n * ctypes.py_object)()
        for i in range(n):
            arr[i] = None
        return arr



myDict = HashTable()

''' See the test below for performance '''

for i in range(20):
    myDict[i] = i
    

del myDict[12]
del myDict[0]
del myDict[11]

#A tuple can be created from every object that is iterable
myTuple = tuple(myDict)

print(myTuple)
print(len(myTuple))

