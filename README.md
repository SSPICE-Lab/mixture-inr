# Implicit Neural Representation-based Image Compression using Mixture Networks

This repository contains the source code for the paper "Breaking the Barriers of One-to-One Usage of Implicit Neural Representation in Image Compression: A Linear Combination Approach with Performance Guarantees". The paper is under review at the IEEE Internet of Things Journal.

If you have any questions, please feel free to contact us at syerragu@buffalo.edu.

## Abstract

In an era where the exponential growth of image data driven by the Internet of Things (IoT) is outpacing traditional storage solutions, this work explores and advances the potential of Implicit Neural Representation (INR) as a transformative approach to image compression. INR leverages the function approximation capabilities of neural networks to represent various types of data. While previous research has employed INR to achieve compression by training small networks to reconstruct large images, this work proposes a novel advancement: representing multiple images with a single network. By modifying the loss function during training, the proposed approach allows a small number of weights to represent a large number of images, even those significantly different from each other. A thorough analytical study of the convergence of this new training method is also carried out, establishing upper bounds that not only confirm the method's validity but also offer insights into optimal hyperparameter design. The proposed method is evaluated on the Kodak, ImageNet, and CIFAR-10 datasets. Experimental results demonstrate that all 24 images in the Kodak dataset can be represented by linear combinations of two sets of weights, achieving a peak signal-to-noise ratio (PSNR) of 26.5 dB with as low as 0.2 bits per pixel (BPP). The proposed method matches the rate-distortion performance of state-of-the-art image codecs, such as BPG, on the CIFAR-10 dataset. Additionally, the proposed method maintains the fundamental properties of INR, such as arbitrary resolution reconstruction of images.

## Requirements

The code is implemented in Python 3.10 with the following dependencies:

- torch 2.3.1
- numpy 1.26.3
- torchvision 0.18.1
- cv2 4.9.0
- rich 13.7.1 (for pretty printing tables)

## Datasets

The datasets used are:

- Kodak dataset: http://r0k.us/graphics/kodak/
- ImageNet dataset: http://image-net.org/
- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

## Usage

The base model for the proposed method is implemented in `mixtures/mixture_network.py`. The training script for the datasets is implemented in `<dataset name>_simulations.py`. The scripts can be run as follows:

```bash
python kodak_simulations.py
python imagenet_simulations.py
python cifar10_simulations.py
```

Each run will have a run name associated with it based on the hyperparameters used. The run name will be printed to the console. The generated images will be saved in the `test_images/<run name>` directory.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
