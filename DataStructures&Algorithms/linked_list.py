
"""
    This is a very simple LinkedList implementation
"""

class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.header = Node(None)
        self.size = 0

    def __len__(self):
        return self.size
    
    def addFront(self, val):
        currFirst = self.header.next
        self.header.next = Node(val, currFirst)
        self.size += 1

    def find(self, val):
        cursor = self.header.next
        while(cursor != None):
            if(cursor.val == val):
                return cursor
            cursor = cursor.next
        return None
    
    def remove(self, val):
        cursor = self.header
        while(cursor.next != None):
            if(cursor.next.val == val):
                cursor.next = cursor.next.next #GC will do the deletion 
                self.size -= 1
                return
            cursor = cursor.next
        raise ValueError("{} is not in the linked list".format(val))
    
    def reverseHelper(self, node):
        if(node.next == None):
            return node
        self.reverseHelper(node.next).next = node
        return node

    def reverse(self):
        if(self.header.next == None):
           return
        newFirstNode = self.header.next
        while (newFirstNode.next != None):
            newFirstNode = newFirstNode.next
        self.reverseHelper(self.header.next).next = None #returns the pre-reverese() first node
        self.header.next = newFirstNode

    def __next__(self):
        if(self.cursor == None):
            raise StopIteration()
        node = self.cursor
        self.cursor = self.cursor.next
        return node

    def __iter__(self):
        self.cursor = self.header.next
        return self
    

myLList = LinkedList()

for i in range(10):
    myLList.addFront(i)

for i in range(3):
    myLList.remove(i)

#myLList.remove(30) value error as expected

for node in myLList:
    print(node.val)

print()

# print(len(myLList))

# print(myLList.find(6).val)

myLList.reverse()

for node in myLList:
    print(node.val)
