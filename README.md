# A multi-agentic framework for real-time, autonomous freeform metasurface design

**Publication out now in** [Science Advances](https://www.science.org/doi/10.1126/sciadv.adx8006)

This repository hosts the resources accompanying our study, **[A multi-agentic framework for real-time, autonomous freeform metasurface design](https://www.science.org/doi/10.1126/sciadv.adx8006)**.

## Overview

We present *MetaChat*, a multi-agentic computer-aided design framework, which combines agency with millisecond-speed deep learning surrogate solvers to automate and accelerate photonics design. MetaChat is capable of performing complex freeform design tasks in nearly real-time, as opposed to the days-to-weeks required by the manual use of conventional computing methods and resources.

Near real-time, multi-objective, multi-wavelength autonomous metasurface design is enabled by two key contributions:
- **Agentic Iterative Monologue (AIM):** *A novel agentic system designed to seamlessly automate multiple-agent collaboration, human-designer interaction, and computational tools*
- **FiLM WaveY-Net:** *A semi-general fullwave surrogate solver, which supports conditional fullwave modeling—enabling simulations with variable conditions, including source angle, wavelength, material, and device topology—while maintaining high fidelity to the governing physics*

![MetaChat framework overview](figs/fig1.png)

## Repository structure

- `metachat-aim/`: Source code for the AIM agentic design stack.
- `film-waveynet/`: Source code for the FiLM WaveY-Net surrogate solver, including scripts for training and inference (pretrained weights downloadable via [Zenodo](https://zenodo.org/records/15802727), training and validation data downloadable via [Stanford Digital Repository](https://purl.stanford.edu/dq123fg9049); see below).

## Data availability

All data used for training and validation in the study and referenced by the code here (dielectric structures, sources, Ex, Ey, and Hz fields) can be downloaded from the [Stanford Digital Repository](https://purl.stanford.edu/dq123fg9049). Further information can be found on the [Metanet Page](http://metanet.stanford.edu/search/metachat/). The pretrained `best_model.pt` checkpoint is hosted on [Zenodo](https://zenodo.org/records/15802727).

## Reference

The following BibTeX entry can be used to cite MetaChat, this code, and data:

```
@article{lupoiu2025multiagentic,
	title = {A multi-agentic framework for real-time, autonomous freeform metasurface design},
	volume = {11},
	url = {https://www.science.org/doi/full/10.1126/sciadv.adx8006},
	doi = {10.1126/sciadv.adx8006},
	language = {en},
	number = {44},
	journal = {Science Advances},
	author = {Lupoiu, Robert and Shao, Yixuan and Dai, Tianxiang and Mao, Chenkai and Edée, Kofi and Fan, Jonathan A.},
	year = {2025},
}
```

## Contact

Corresponding author: jonfan@stanford.edu

If you have any questions or need help setting up either AIM or FilM WaveY-Net, don't hesitate to reach out!