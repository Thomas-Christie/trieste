Add (1 - 0.9^num_iters) as a coefficient for updating lambda. Not that the iteration was only incrememented by 1 each batch
for each batch in this run (rather than 3 i.e. more conservatively updating lambda for longer):

- Batches = 3
- Inequality One Initial Lambda = [0.0, 1.0, 2.0]
- Inequality Two Initial Lambda = [0.0, 0.0, 0.0]