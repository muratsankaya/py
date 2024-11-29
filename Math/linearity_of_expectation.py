"""
Let X1, X2 and XN be random variables
Lineariy of Expectation:
E[X1 + X2 + ... + XN] = E[X1] + E[X2] + ... + E[XN]

Note***:
- This holds true as long as rhs is defined.
- This holds true regardless of the dependencies of random variables


Examples demonstrate outcomes of fair dice rolls.
"""

occurances = {}

for i in range(1, 7):
    for j in range(1, 7):
        occurances[i + j] = occurances.get(i + j, 0) + 1

print(occurances)

s = 0
for i in range(2, 13):
    s += i * (occurances[i] / 36)

print(s)


occurances = {}

for i in range(1, 7):
    for j in range(1, 7):
        for k in range(1, 7):
            occurances[i + j + k] = occurances.get(i + j + k, 0) + 1

print(occurances)

s = 0
for i in range(3, 19):
    s += i * (occurances[i] / 216)

print(s)
