Ran TS-AL on toy problem 100 times. Initialised penalty using method in Picheny's paper (which leads to higher
initial penalties) to avoid optimisation issues in later iterations. 

Optimisation failed around run 18 (on the final iteration of BO) so increased `num_recovery_runs` in optimiser
from 10 to 50. This stopped any further optimisation failures from occuring.

Used following parameters:

```
NUM_INITIAL_SAMPLES = 5
BATCH_SIZE = 1
NUM_BO_ITERS = 50
EPSILON = 0.001
```