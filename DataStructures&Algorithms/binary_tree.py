"""
    This is a basic binary tree implementation. Uses a balancing insertion policy
"""

from collections import deque
from random import random


class TreeNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


class BinaryTree:
    def __init__(self, data):
        self.root = TreeNode(data)
        self.size = 0

    def insert(self, data):
        cursor = self.root
        while True:
            if not cursor.left:
                cursor.left = TreeNode(data)
                break
            if not cursor.right:
                cursor.right = TreeNode(data)
                break
            p = random()
            if p < 0.5:
                cursor = cursor.left
            else:
                cursor = cursor.right

    # add a delete node here

    def maxDepth(self):
        def helper(node):
            if not node:
                return 0
            return max(helper(node.left), helper(node.right)) + 1

        return helper(self.root)

    def leavesCount(self):
        def helper(node):
            if not node:
                return 0
            if not node.left and not node.right:
                return 1
            return helper(node.left) + helper(node.right)

        return helper(self.root)

    def printBreadthFirst(self):

        q = deque()
        q.append((self.root, 1))
        currLevel = 1

        while len(q) != 0:
            prev = q.popleft()

            # print newline at each new level
            if prev[1] > currLevel:
                print()
                currLevel = prev[1]

            print(prev[0].data, end=" ")

            if prev[0].left:
                q.append((prev[0].left, prev[1] + 1))

            if prev[0].right:
                q.append((prev[0].right, prev[1] + 1))

    def printPreOrder(self):
        def helper(node):
            if not node:
                return
            print(node.data, end=" ")
            helper(node.left)
            helper(node.right)

        helper(self.root)

    def printInOrder(self):
        def helper(node):
            if not node:
                return
            helper(node.left)
            print(node.data, end=" ")
            helper(node.right)

        helper(self.root)

    def printPostOrder(self):
        def helper(node):
            if not node:
                return
            helper(node.left)
            helper(node.right)
            print(node.data, end=" ")

        helper(self.root)
    
    def printMorrisInorder(self):
        curr, pred = self.root, None
        while curr:
            if curr.left:
                pred = curr.left
                while pred.right and pred.right is not curr:
                    pred = pred.right

                # will happen when we are traversing on the same route a second time
                # so the there will be a thread between the curr and pred
                # which should be removed
                if pred.right:
                    pred.right = None
                    print(curr.data, end=",")

                    # continue traversing the right subtree
                    curr = curr.right

                else:
                    # set the predecessor
                    pred.right = curr

                    # continue traversing on the left sub-tree
                    curr = curr.left

            else:

                # reached the leftmost node of a subtree
                # print curr and continue on the right side
                print(curr.data, end=",")
                curr = curr.right


myBTree = BinaryTree(0)

for i in range(1, 9):
    myBTree.insert(i)

print("size: ", myBTree.size)

myBTree.printBreadthFirst()
# print("pre order: ")
# myBTree.printPreOrder()
# print("in order: ")
# myBTree.printInOrder()
# print("post order: ")
# myBTree.printPostOrder()

print()

print("max depth: {} ".format(myBTree.maxDepth()))

print("leaves count: {} ".format(myBTree.leavesCount()))

myBTree.printInOrder()
print()
myBTree.printMorrisInorder()
