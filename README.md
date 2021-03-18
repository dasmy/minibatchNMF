# beta_nmf_minibatch: NMF with beta-divergence and mini-batch updates

Theano based GPGPU implementation of NMF with beta-diveregence and mini-batch multplicative updates.


## Dependencies

beta_nmf_minibatch need Python >= 2.7, numpy >= 10.1, Theano >= 0.8, scikit-learn >= 0.17.1, h5py >= 2.5, itertools and more_itertools

## Documentation

Documentation available at http://rserizel.github.io/minibatchNMF/


## Getting Started

A short example is available as a [notebook] or classical [python script]

[notebook]: minibatch_BetaNMF_howto.ipynb
[python script]: minibatch_BetaNMF_howto.py

## Citation

If you are using this source code please consider citing the following paper: 

> R. Serizel, S. Essid, and G. Richard. “Mini-batch stochastic approaches for accelerated multiplicative updates in nonnegative matrix factorisation with beta-divergence”. Accepted for publication In *Proc. of MLSP*, p. 5, 2016.

Bibtex
```
	@inproceedings{serizel2016batch,
  	title={Mini-batch stochastic approaches for accelerated multiplicative updates in nonnegative matrix factorisation with beta-divergence},
  	author={Serizel, Romain and Essid, Slim and Richard, Ga{\"e}l},
  	booktitle={IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  	pages={5470--5474},
  	year={2016},
  	organization={IEEE}
	}
```

## Author

Romain Serizel, 2014 -- Present

## License

Copyright 2014-2016 Romain Serizel

This software is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt)
