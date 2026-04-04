from concurrent.futures import ThreadPoolExecutor
from itertools import batched
import torch as t


def main():
    print("Hello from canonical-interp!")


if __name__ == "__main__":
    a = t.arange(10)

    def square(idxs):
        a[idxs] *= a[idxs]

    print(a)
    with ThreadPoolExecutor(max_workers=100) as executor:
        executor.map(square, list(map(list, batched(range(10), 2))))

    print(a)
