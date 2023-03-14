Goal of Experiment - See how KKT algorithm performs with additional minor details from the paper.

Conclusion - Relaxing assumption about constraints being binding isn't going to help, as we will then just choose points which
slightly violate the constraint, which will have higher EI.

Used same parameters as in paper:
- EI_EPSILON = 0.001
- INITIAL_ALPHA = 0.2
- ALPHA_LOWER_BOUND = 0.01
- INITIAL_SAMPLES = 6
- NUM_BO_ITERS = 20

Note that BO now sometimes terminates early.

Used default parameters for continuous optimiser.
