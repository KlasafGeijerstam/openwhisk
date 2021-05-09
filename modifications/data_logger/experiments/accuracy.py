from sys import stdin

def p(a, b):
    print(f"\"{str(round(a/b, 2)).replace('.', ',')}\"")


input()

for line in stdin:
    estimation, actual = map(int, line.strip().split(","))

    if estimation > actual:
        p(actual,estimation)
    else:
        p(estimation,actual)