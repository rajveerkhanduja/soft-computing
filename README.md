# soft-computing

##Write a Python program to implement all steps of GA (selection, crossover,

import random

# fitness function
def fitness(x):
    return x * x        # maximize x²

# initial population
pop = [random.uniform(-10, 10) for _ in range(5)]

for gen in range(10):
    new_pop = []

    for _ in range(5):
        # selection
        p1 = random.choice(pop)
        p2 = random.choice(pop)

        # crossover
        child = (p1 + p2) / 2

        # mutation
        child += random.uniform(-1, 1)

        new_pop.append(child)

    pop = new_pop
    best = max(pop, key=fitness)
    print("Gen", gen+1, " Best:", best)

