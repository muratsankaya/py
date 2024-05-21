import random

def quick_sort_retry(arr):
    """
    A better version of randomized Quicksort
    Wort Case runtime: O(n^2)
    Expected Case runtime: O(nlogn)
    High probability bound: O(nlogn)
    
    The key idea is that the while loop
    is expected to run 2 times.
    """
    n = len(arr)

    if n <= 1:
        return arr
    
    while True:
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

        if len(left) >= n // 4 and len(right) >= n // 4:
            break

    left = quick_sort_retry(left)
    right = quick_sort_retry(right)
    return  left + [pivot] + right



def rand_quick_sort(arr):
    """
    A worse version of randomized QuickSort
    Wort Case runtime: O(n^2)
    Expected Case runtime: O(n^2)
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
for i in range(25):
    arr.append(random.randint(-100, 100))

# print(f"Randomly created array is {arr}")
# print(f"Sorted array is {rand_quick_sort(arr)}")

print(f"Randomly created array is {arr}")
print(f"Sorted array is {quick_sort_retry(arr)}")

