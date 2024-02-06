
from typing import List
import random
from timeit import default_timer as timer

def merge(l : List[int], left : tuple[int], right : tuple[int]):
    
    leftList, rightList = l[left[0]:left[1]+1], l[right[0]:right[1]+1]

    i, j = 0, 0

    for k in range(left[0], right[1]+1):
        if(i < len(leftList) and j < len(rightList)):
            if(leftList[i] <= rightList[j]):
                l[k] = leftList[i]
                i += 1
            else:
                l[k] = rightList[j]
                j += 1
        elif(i < len(leftList)):
            l[k] = leftList[i]
            i += 1
        else:
            l[k] = rightList[j]
            j += 1

        

def mergeSort(l : List[int], low: int, high: int): #takes the first and the last index if "l" as low and high 
    if(high  == low):
        return (low, high)
    
    mid = low//2 + high//2

    left = mergeSort(l, low, mid)
    right = mergeSort(l, mid+1, high)
    merge(l, left, right)
    return (low, high)


# nums = random.sample(range(1, 101), 100)
# mergeSort(nums, 0, len(nums) - 1)
# print(nums)

# Test Case 2: Random list of 1000 elements
nums = random.sample(range(-1000000, 1000000), 1000000)

start = timer()
mergeSort(nums, 0, len(nums) - 1)
end = timer()

print(end - start)

nums = random.sample(range(-1000000, 1000000), 1000000)

start = timer()
nums.sort()
end = timer()

print(end - start)

# # Test Case 3: Sorted list of 1000 elements in descending order
# nums = list(range(1000, 0, -1))
# mergeSort(nums, 0, len(nums) - 1)
# print(nums)

# # Test Case 4: Sorted list of 10000 elements in ascending order
# nums = list(range(1, 10001))
# mergeSort(nums, 0, len(nums) - 1)
# print(nums)

