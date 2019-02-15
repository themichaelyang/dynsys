import math

def f(x):
    return math.e**x - 2

def g(x):
    return math.log(x + 2)

def main():
    x = 0
    for i in range(10):
        x = f(x)
        print(x)
    print()
    x = 0
    for i in range(10):
        x = g(x)
        print(x)

main()
