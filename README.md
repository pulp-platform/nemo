# NEMO (NEural Minimizer for pytOrch)
**NEMO (NEural Minimizer for pytOrch)** is a small library for minimization of Deep Neural Networks developed in PyTorch, aimed at their deployment on ultra-low power, highly memory constrained platforms.
NEMO features include:
 - deployment-related transformations such as BatchNorm folding, bias removal, weight equalization 
 - collection of statistics on activations and weights
 - post-training quantization
 - quantization-aware fine-tuning, with partially automated precision relaxation
 - mixed-precision quantization
 - bit-accurate deployment model
 - export to ONNX

NEMO operates on three different "levels" of quantization-aware DNN representations, all built upon `torch.nn.Module`s and `torch.autograd.Function`s:
 - fake-quantized *FQ*: replaces regular activations (e.g., ReLU) with quantization-aware ones (PACT) and dynamically quantized weights (with linear PACT-like quantization), maintaining full trainability (similar to the native PyTorch support, but not based on it).
 - quantized-deployable *QD*: replaces all function with deployment-equivalent versions, trading off trainability for a more accurate representation of numerical behavior on real hardware.
 - integer-deployable *ID*: replaces all activation and weight tensors used along the network with integer-based ones. It aims at bit-accurate representation of actual hardware behavior.
All the quantized representations support mixed-precision weights (signed and asymmetric) and activations (unsigned). The current version of NEMO targets per-layer quantization; work on per-channel quantization is in progress.

NEMO is organized as a Python library that can be applied with relatively small changes to an existing PyTorch based script or training framework.

# Example
The `examples` directory includes fully commented examples for training on MNIST, CIFAR-10.

# License
NEMO is released under Apache 2.0, see the LICENSE file in the root of this repository for details.

# Requirements
The NEMO library (NEural Minimizer for tOrch) has only minimal dependencies on PyTorch 1.3 and TensorboardX. To set up a minimum working environment in Anaconda/Miniconda, run

```
conda create -n nemo python=3
source activate nemo
conda install -y pytorch torchvision -c pytorch
conda install -y scipy
pip install tensorboardX
```

# Acknowledgements
![ALOHA Logo](/var/aloha.png)

NEMO is an outcome of the European Commission Horizon 2020 ALOHA Project, funded under the EU's Horizon 2020 Research and Innovation Programme, grant agreement no. 780788.
