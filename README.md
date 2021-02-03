# Learning of Deep learning with PyTorch

## Image classification

- **lenet-5.py** - original version of LeNet-5 (for the MNIST data)
- **image-classification.py** - an adapted version of LeNet-5 fot the CIFAR10 dataset
- **alexnet.py** - an adapted version of AlexNet fot the CIFAR10 dataset

## Embedding

- **autoencoder.py** - basic autoencoder for the MNIST dataset
- **skip-gram.py** - word2vec based on Skip-Gram

## Sequencial analysis

- **seq2one.py** - simple RNN for the QRSU classification task with with a heatmap for model interpretability
- **seq2one-lstm.py** - RNN with LSTM for the QRSU classification task with heatmaps for model interpretability

Requires (for seq2one.py, seq2one-lstm.py):

- wget https://github.com/Atcold/pytorch-Deep-Learning/blob/master/res/sequential_tasks.py
- wget https://github.com/Atcold/pytorch-Deep-Learning/blob/master/res/plot_lib.py

Tested on: Python 3.7, torch 1.5.1 (+CUDA 9.2)

Based on tutorials:

- https://www.oreilly.com/library/view/fundamentals-of-deep/9781491925607
- https://dl-lab.eu
- https://atcold.github.io/pytorch-Deep-Learning