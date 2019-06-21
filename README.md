# Simplified Probabilistic Neural Programmed Network

In this repository, I removed the Tranform, Combine, and Describe modules along with Gated Product of Experts, and replaced them by a one-hot encoding vectors representing object attributes. I got similar results to the original PNP paper. I haven't been able to find an advantage over the proposed method by:

[Probabilistic Neural Programmed Networks for Scene Generation](http://www2.cs.sfu.ca/~mori/research/papers/deng-nips18.pdf)




## Contents
1. [Overview](#overview)
2. [Environment Setup](#environment)
3. [Data and Pre-trained Models](#data-and-models)
	- [CLEVR-G](#CLEVR-G)
4. [Configurations](#configurations)
5. [Code Guide](#code-guide)
	- [Neural Operators](#neural-operators)
6. [Training Model](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)


## Environment

All code was tested on Ubuntu 16.04 with Python 2.7 and **PyTorch 0.4.0** (but the code should also work well with Python 3). To install required environment, run:

```bash
pip install -r requirements.txt   
```

For running our measurement (a semantic correctness score based on detector), check [this submodule](https://github.com/woodfrog/SemanticCorrectnessScore/tree/483c6ef2e0548fcc629059b84c489cd4e0c19f86) for full details (**it's released now**).

## Data and Models

### CLEVR-G

Dataset:
[64x64 CLEVR-G](https://drive.google.com/open?id=10yP0ki9EqxOacCL08mDQiDbVvUeO8m41).
Please download and zip it into **./data/CLEVR** if you want to use it with our model.

### COLOR-MNIST

- [PNP-Net COLOR-MNIST 64x64](https://www.dropbox.com/s/jf5u7yosyisf8zd/MULTI_8000_64_100.tar?dl=0)



## Configurations

[global configuration files](configs/pnp_net_configs.yaml) is used to set up all configs, including the training settings and model hyper-parameters.


## Code Guide

### Neural Operators

The core of PNP-Net is a set of **neural modular operators**. A brief introduction by the authers are here:


- **Concept Mapping Operator**


<div align='center'>
  <img src='images/mapping.png' width='256px'>
</div>


Convert one-hot representation of word concepts into appearance and scale distribution. 
[code](lib/modules/ConceptMapper.py)
	
- **Combine Operator**

<div align='center'>
  <img src='images/combine.png' width='256px'>
</div>


Combine module combines the latent distributions of two attributes. [code](lib/modules/Combine.py)


- **Describe Operator**

<div align='center'>
  <img src='images/describe.png' width='256px'>
</div>

Attributes describe an object, this module takes the distributions of attributes (merged using combine module) and uses it to render the distributions of an object. [code](lib/modules/Describe.py)
	
	
- **Transform Operator**

<div align='center'>
  <img src='images/transform.png' width='256px'>
</div>
	
This module first samples a size instance from an object's scale distribution and then use bilinear interpolation to re-size the appearance distribution. [code](lib/modules/Transform.py)
	

- **Layout Operator**

<div align='center'> 
  <img src='images/layout.png' width='256px'>
</div> 


Layout module puts latent distributions of two different objects (from its children nodes) on a background latent canvas according to the offsets of the two children objects. [code](models/PNPNet/pnp_net.py#L267)



## Training

The default training of the SIMPLE net can be started by: 

```bash
python mains/pnpnet_main.py --config_path configs/simplified_pnp_net_configs.yaml
```

The default training of the PNP net can be started by: 

```bash
python mains/pnpnet_main.py --config_path configs/pnp_net_configs.yaml
```

Make sure that you are in the project root directory when typing the above command. 




## Evaluation

The evaluation has two major steps:

1. Generate images according to the semantics in the test set using pre-trained model. 

2. Run our [detector-based semantic correctness score](https://github.com/woodfrog/SemanticCorrectnessScore) to evaluate the quality of images. Please check that repo for more details about our proposed metric for measuring semantic correctness of scene images.


For generating test images using pre-trained model, first set the code mode to be **test**, then set up the checkpoint path properly in the config file, finally run the same command as training:

```bash
python mains/pnpnet_main.py --config_path configs/pnp_net_configs.yaml
```


## Results

When the scene becomes too complex, PNP-Net can suffer from the following problems:

1. It might fail to handle occlusion between objects. When multiple objects overlap, their latents get mixed on the background latent canvas, and the appearance of objects can be distorted.

2. It might put some of the objects out of the image boundary, therefore some images do not contain the correct number of objects as described by the semantics.
