# Differential Surfel Rasterization

This project extends the [surfel-rasterization engine](https://github.com/hbb1/diff-surfel-rasterization) of 2D Gaussian Splatting, by integrating a relocation kernel based on [3DGS-MCMC](https://ubc-vision.github.io/3dgs-mcmc/), absolute view-space gradient based on [AbsGS](https://github.com/TY424/AbsGS), and gaussian trimming based on [TrimGS](https://github.com/YuxueYang1204/TrimGS)

This relocation strategy enhances handling Gaussian splat parameters, focusing on maintaining sample state probabilities during heuristic moves like 'move', 'split', 'clone', 'prune', and 'add'.

## Installation 
To use the engine, follow these steps:

- Clone the repository:
```bash
git clone https://github.com/hugoycj/diff-surfel-rasterization.git
cd diff-surfel-rasterization
```
- Install the package
```bash
pip install . --no-cache
```
- Alternatively, you can set up the Python C++ extension project:
```bash
python setup.py build_ext --inplace
```

## Example Usage
```python
from diff_surfel_rasterization import compute_relocation
import torch
import math

N_MAX = 51
BINOMS = torch.zeros((N_MAX, N_MAX)).float().cuda()
for n in range(N_MAX):
    for k in range(n+1):
        BINOMS[n, k] = math.comb(n, k)

def compute_relocation_cuda(
    opacities,  # [N]
    scales,  # [N, 2]
    ratios,  # [N]
):
    """
    Computes new opacities and scales using the MCMC relocation kernel.

    Args:
        opacities (torch.Tensor): Array of opacities for each Gaussian splat.
        scales (torch.Tensor): Array of scales for each Gaussian splat.
        ratios (torch.Tensor): Array of ratios used in relocation computation.

    Returns:
        new_opacities (torch.Tensor): Updated opacities after relocation.
        new_scales (torch.Tensor): Updated scales after relocation.
    """
    N = opacities.shape[0]
    opacities = opacities.contiguous()
    scales = scales.contiguous()
    ratios.clamp_(min=1, max=N_MAX)
    ratios = ratios.int().contiguous()

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, BINOMS, N_MAX
    )
    return new_opacities, new_scales
```

## Acknowledgments

This project builds upon the research and implementations detailed in the following papers:
- [3D Gaussian Splatting as Markov Chain Monte Carlo](https://ubc-vision.github.io/3dgs-mcmc/)
- [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)
