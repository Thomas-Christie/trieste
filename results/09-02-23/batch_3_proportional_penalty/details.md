Toy problem with:
- BS = 3
- Initial Lambda 1 = [1, 2, 3]
- Initial Lambda 2 = [0, 0, 0]
- Initial Penalty = [0.5, 0.5, 0.5]

Updated Penalty - If N ALs in a batch are violated, then for each of these ALs, updated_penalty = original_penalty/(2^N).
