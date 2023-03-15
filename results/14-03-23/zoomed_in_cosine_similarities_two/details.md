Goal of Experiment - Visualise cosine similarities in zoomed-in area around optimal point (same as other experiment on
this date, but including marker at optimal point).

Conclusion - We can see upon zooming in alot that the cosine similarity is indeed ~1 near the optimal point.
However, in later iterations this area of high cosine similarity becomes exceedingly small, which makes
the acquisition function very difficult to optimise.

Used same parameters as in paper:
- EI_EPSILON = 0.001
- INITIAL_ALPHA = 0.2
- ALPHA_LOWER_BOUND = 0.01
- INITIAL_SAMPLES = 6
- NUM_BO_ITERS = 20

Note that BO now sometimes terminates early.

Used default parameters for continuous optimiser.
