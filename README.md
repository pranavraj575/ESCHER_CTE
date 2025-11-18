# ESCHER-CTE: version of [ESCHER](https://github.com/Sandholm-Lab/ESCHER) that updates reference policies

ESCHER is an unbiased model-free method for finding approximate Nash equilibria in large two-player zero-sum games that does not require any importance sampling. ESCHER is principled and is guaranteed to converge to an approximate Nash equilibrium with high probability in the tabular case. 

ESCHER performs outcome sampling Monte-Carlo CFR, which normally produces samples that must be inverse weighted (by the player's reach probability) to be unbiased.
However, ESCHER keeps the _reference strategy_ (used by the player doing an update) fixed, so the reach probability for a particular infoset will be the same across iterations, and the weights can be ignored (if the regret minimizer at each infoset is scale-invariant).

A downside of keeping the reference strategy fixed is that in late iterations, ESCHER does not at all utilize the knowledge of which actions are good/should never be played to improve each player's reference strategy.
To fix this, ESCHER-CTE simply restarts the ESCHER algortihm mid-run, changing each player's reference strategy to be closer to the produced policy.
All regrets are forgotten and must be relearned from scratch (as the scaling will be different due to the change in reference strategy).
It is potentially possible to warm-start the regret minimizers to retain some knowledge of the previous policy.

Requires [open_spiel](https://github.com/deepmind/open_spiel) and, optionally, [ray.](https://github.com/ray-project/ray). 
Instructions for installing both are [here.](https://github.com/indylab/nxdo/blob/master/docs/install.md)

<img src="logo.jpg" border="1"/>

For more information about ESCHER, please see [the paper.](https://arxiv.org/abs/2206.04122)

Tested with python 3.11