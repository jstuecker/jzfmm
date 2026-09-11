# Attribution

jz-fmm was written by Jens Stücker in the Computational
Cosmology group at the University of Vienna. This research was funded in whole
or in part by the Austrian Science Fund (FWF) [10.55776/ESP705]. Development
and benchmarking used the EuroHPC supercomputer LEONARDO, hosted by CINECA.

The source code is available under the MIT license. If you publish work that uses jz-fmm, please cite the accompanying paper:

Jens Stücker and Oskar Foldal (2026), [*JZ-FMM: GPU-native differentiable N-body simulations with the Fast Multipole Method*](https://arxiv.org/abs/2609.09307), arXiv:2609.09307.

```bibtex
@misc{stuecker_foldal_2026_jzfmm,
    author = {St{\"u}cker, Jens and Foldal, Oskar},
    title = {{JZ-FMM}: {GPU}-native differentiable {N}-body simulations with the {Fast Multipole Method}},
    year = {2026},
    eprint = {2609.09307},
    archivePrefix = {arXiv},
    primaryClass = {astro-ph.CO},
    url = {https://arxiv.org/abs/2609.09307}
}
```

Optionally, you may also want to cite the [jz-tree paper (arXiv:2604.05885)](https://arxiv.org/abs/2604.05885), which describes the underlying GPU-native tree algorithms used by jz-fmm:

```bibtex
@misc{stuecker_2026_jztree,
    author = {St{\"u}cker, Jens and Hahn, Oliver and Winkler, Lukas and Gutierrez Adame, Adrian and Fl{\"o}ss, Thomas},
    title = {{JZ-Tree}: {GPU} friendly neighbour search and friends-of-friends with dual tree walks in {JAX} plus {CUDA}},
    year = {2026},
    eprint = {2604.05885},
    archivePrefix = {arXiv},
    primaryClass = {cs.DC},
    url = {https://arxiv.org/abs/2604.05885}
}
```

Questions and feature requests may be submitted through the
[GitHub issue tracker](https://github.com/jstuecker/jzfmm/issues).
