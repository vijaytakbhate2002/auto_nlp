import sys
paths = sys.path
for path in paths:
    print(path)

from .trainer import Trainer