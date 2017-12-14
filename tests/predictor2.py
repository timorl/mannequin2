
import sys
import numpy as np
from mannequin import SimplePredictor
from test_setup import timer

def func(x, y):
    return [np.exp(x), x * y]

pred = SimplePredictor(2, 2)
for _ in range(500):
    xy = np.random.randn(256).reshape(-1, 2)
    pred.sgd_step(xy, [func(x, y) for x, y in xy], lr=0.02)

errors = []
for x in np.linspace(-2.0, 2.0, 21):
    for y in np.linspace(-2.0, 2.0, 21):
        errors.append(np.abs(pred.predict([x, y]) - func(x, y)))
errors = np.mean(errors, axis=0)
sys.stderr.write("Mean errors: %.4f %.4f\n" % tuple(errors))
assert (errors < 0.2).all()

assert timer() < 1.9