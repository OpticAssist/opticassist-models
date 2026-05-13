import sys
from typing import Literal


# Prints error output to be collected in Rust
def eprint(*values: object, sep: str | None = " ",
           end: str | None = "\n",
           flush: Literal[False] = False):
    print(*values, sep=sep, end=end, flush=flush, file=sys.stderr)


def main():
    print("hello, world!")

if __name__ == '__main__':
    main()
