
import random

def rand_quick_sort(arr):
    """
    Wort Case runtime O(n^2)
    Expected Case runtime O(nlogn)
    """
    n = len(arr)

    if n <= 1:
        return arr
    
    pivot_index = random.randint(0, n-1)
    pivot = arr[pivot_index]
    left, right = [], []
    for i in range(n):
        if i == pivot_index:
            continue

        if arr[i] <= pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])

    left = rand_quick_sort(left)
    right = rand_quick_sort(right)
    return  left + [pivot] + right


arr = []
for i in range(40):
    arr.append(random.randint(-100, 100))

print(f"Randomly created array is {arr}")
print(f"Sorted array is {rand_quick_sort(arr)}")

