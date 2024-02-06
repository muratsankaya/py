

""" 
    This is an overly simple ArrayList implemtation of Python. It does not 
    support arithmatic operations and many more capabilities that python list provides.
"""

import random
import ctypes


class ArrayList:
    def __init__(self):
        self.capacity = 2
        self.size = 0
        self.arr = self.makeArray(self.capacity)

    def __getitem__(self, i):
        if(i < 0):
            i += self.size
        if(i < 0 or i >= self.size):
            raise IndexError("IndexError: ArrayList index out of range ")
        return self.arr[i]
    
    def __setitem__(self, i, data):
        if(i < 0):
            i += self.size
        if(i < 0 or i >= self.size):
            raise IndexError("IndexError: ArrayList index out of range ")
        self.arr[i] = data

    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if(self.i == self.size):
            raise StopIteration
        data = self.arr[self.i]
        self.i += 1
        return data
    
    def __len__(self):
        return self.size
    
    def __contains__(self, data):
        for i in range(self.size):
            if(self.arr[i] == data):
                return True
        return False

    def makeArray(self, n):
        return (n * ctypes.py_object)()
    
    def resize(self, shrink):
        if(shrink):
            self.capacity //= 2
        else:
            self.capacity *= 2
        
        newArray = self.makeArray(self.capacity)

        for i in range(self.size):
            newArray[i] = self.arr[i]

        self.arr = newArray #garbage collector will do the deletion


    def append(self, data):
        if(self.size == self.capacity):
            self.resize(False)
        self.arr[self.size] = data
        self.size += 1


    def pop(self, i=-1):
        if(i < 0):
            i += self.size
        if(i < 0 or i >= self.size):
            raise IndexError("IndexError: ArrayList index out of range ")

        data = self.arr[i]

        if(self.size == self.capacity//4):
            self.resize(True)

        for j in range(i, self.size-1):
                self.arr[j] = self.arr[j+1]
        
        self.size -= 1
        return data

    def insert(self,i, data):
        if(i < 0):
            i += self.size
        if(i < 0 or i >= self.size):
            raise IndexError("IndexError: ArrayList index out of range ")
        
        if(self.size == self.capacity):
            self.resize(False) #resizing could be optimized but not necesarry for this implementation

        for j in range(self.size, i, -1):
            self.arr[j] = self.arr[j-1]

        self.size += 1   
        self.arr[i] = data 

    
# data = ArrayList()

# capacity, count, p, n = 0, 0, 0, 0

# ''' See the test below for performance '''

# print("newtest")
# while(count < 85):

#     p = random.random()
#     n = random.randrange(1, 201)
#     for i in range(n):
#         if(p < .5 or len(data) < n ):
#             data.append(1)
#         else:
#             data.pop()

#         if(data.capacity != capacity):
#             capacity = data.capacity
#             print("capacity: {}, size: {}".format(capacity, len(data)))

        
#     count += 1

# """ See the simple tests below to test the methods """

# data.pop()

# for i in range(10):
#     data.append(i)

# data.pop()

# for elem in data:
#     print(elem, end=" ")

# print()

# print(data[-3])

# print(data.pop(2))

# data.insert(4, 50)

# data.insert(-3, 60)

# data[-7] = 40

# print(len(data))

# for elem in data:
#     print(elem, end=" ")
# print()


#print(data[-10])

# for elem in data:
#     print(elem, end=" ")

# print()
        
    