
from timeit import default_timer as timer

def fib(n):
    if(n < 0):
        return -1
    if(n == 0):
        return 0
    if(n == 1):
        return 1
    return fib(n-1) + fib(n-2)


def memoFib(n, arr):
    if(n < 0):
        return -1
    if(n == 0 or n == 1):
        return 1
    if( arr[n] > 0 ):
        return arr[n]
    
    arr[n] = memoFib(n-1, arr) + memoFib(n-2, arr)
    return arr[n]


def printFib(n):
    arr = [0]*(n+1)

    # start = timer()

    # print(fib(n))
    
    # end = timer()

    # print()
    # print(end - start)

    start = timer()

    for i in range(n):
        print(memoFib(i, arr), end=" ");
        arr = [0]*(n+1)

    end = timer()

    print()
    print(end - start)


printFib(15);
