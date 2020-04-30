# NEMO (NEural Minimizer for pytOrch)
**NEMO (NEural Minimizer for pytOrch)** is a small library for minimization of Deep Neural Networks developed in PyTorch, aimed at their deployment on ultra-low power, highly memory constrained platforms, in particular (but not exclusively) PULP-based microcontrollers.
NEMO features include:
 - deployment-related transformations such as BatchNorm folding, bias removal, weight equalization 
 - collection of statistics on activations and weights
 - post-training quantization
 - quantization-aware fine-tuning, with partially automated precision relaxation
 - mixed-precision quantization
 - bit-accurate deployment model
 - export to ONNX

NEMO operates on three different "levels" of quantization-aware DNN representations, all built upon `torch.nn.Module` and `torch.autograd.Function`:
 - fake-quantized *FQ*: replaces regular activations (e.g., ReLU) with quantization-aware ones (PACT) and dynamically quantized weights (with linear PACT-like quantization), maintaining full trainability (similar to the native PyTorch support, but not based on it).
 - quantized-deployable *QD*: replaces all function with deployment-equivalent versions, trading off trainability for a more accurate representation of numerical behavior on real hardware.
 - integer-deployable *ID*: replaces all activation and weight tensors used along the network with integer-based ones. It aims at bit-accurate representation of actual hardware behavior.
All the quantized representations support mixed-precision weights (signed and asymmetric) and activations (unsigned). The current version of NEMO targets per-layer quantization; work on per-channel quantization is in progress.

NEMO is organized as a Python library that can be applied with relatively small changes to an existing PyTorch based script or training framework.

# Installation and requirements
The NEMO library currently supports PyTorch >= 1.3.1 and runs on Python >= 3.5.
To install it from PyPI, just run:
```
pip install pytorch-nemo
```
You can also install a development (and editable) version of NEMO by directly downloading this repo:
```
git clone https://github.com/pulp-platform/nemo
cd nemo
pip install -e .
```
Then, you can import it in your script using
```
import nemo
```

# Example
- MNIST post-training quantization: https://colab.research.google.com/drive/1AmcITfN2ELQe07WKQ9szaxq-WSu4hdQb

# Documentation
Full documentation for NEMO is under development (see `doc` folder). You can find a technical report covering the deployment-aware quantization methodology here: https://arxiv.org/abs/2004.05930

# License
NEMO is released under Apache 2.0, see the LICENSE file in the root of this repository for details.

# Acknowledgements
![ALOHA Logo](/var/aloha.png)

NEMO is an outcome of the European Commission [Horizon 2020 ALOHA Project](https://www.aloha-h2020.eu/), funded under the EU's Horizon 2020 Research and Innovation Programme, grant agreement no. 780788.
