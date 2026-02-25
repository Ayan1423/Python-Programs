import numpy as np

numbers = np.arange(1, 101)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

primes = np.array([num for num in numbers if is_prime(num)])

print("Prime numbers between 1 and 100:")
print(primes)