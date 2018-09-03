# Numpy-ELM

## Overview

<div align="center">
    <img src="https://i.imgur.com/YdgQOlH.png" width=600>
</div>

In this repository, we provide a keras-like numpy implementation of Extreme Learning Machine (ELM)
introduced by Huang et al. in this [paper](http://ieeexplore.ieee.org/document/1380068/?reload=true).

ELM is a neural-network-based new learning schieme which can learn fast.
The training will always converge to the global optimal solution,
while ordinary backpropagation-based neural networks have to deal with
the local minima problem.

## Dependencies

We tested our codes using the following libraries.

* Python 3.6.0
* Numpy 1.14.1
* Keras (tensorflow backend) 2.1.5
* Tensorflow 1.7.0

We used Keras only for downloading the MNIST dataset.

You don't have to use exactly the same version of the each library,
but we can not guarantee the codes work well in the case.

All the above libraries can be installed in the following command.

`$ pip install -U numpy Keras tensorflow`

## Usage

Here, we show how to train an ELM model and predict on it.
For the sake of simplicity, we assume to train the model on MNIST, a
hand-written digits dataset.

```python
from keras.datasets import mnist
from keras.utils import to_categorical
from elm import ELM, load_model
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden_nodes', type=int, default=1024)
parser.add_argument('--loss',
    choices=['mean_squared_error', 'mean_absolute_error'],
    default='mean_squared_error',
)
parser.add_argument('--activation',
    choices=['sigmoid', 'identity'],
    default='sigmoid',
)

def softmax(x):
    c = np.max(x, axis=-1)
    upper = np.exp(x - c)
    lower = np.sum(upper, axis=-1)
    return upper / lower

def main(args):
    # ===============================
    # Load dataset
    # ===============================
    n_classes = 10
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    # ===============================
    # Preprocess
    # ===============================
    x_train = x_train.astype(np.float32) / 255.
    x_train = x_train.reshape(-1, 28**2)
    x_test = x_test.astype(np.float32) / 255.
    x_test = x_test.reshape(-1, 28**2)
    t_train = to_categorical(t_train, n_classes).astype(np.float32)
    t_test = to_categorical(t_test, n_classes).astype(np.float32)

    # ===============================
    # Instantiate ELM
    # ===============================
    model = ELM(
        n_input_nodes=28**2,
        n_hidden_nodes=args.n_hidden_nodes,
        n_output_nodes=n_classes,
        loss=args.loss,
        activation=args.activation,
        name='elm',
    )

    # ===============================
    # Training
    # ===============================
    model.fit(x_train, t_train)
    train_loss, train_acc = model.evaluate(x_train, t_train, metrics=['loss', 'accuracy'])
    print('train_loss: %f' % train_loss)
    print('train_acc: %f' % train_acc)

    # ===============================
    # Validation
    # ===============================
    val_loss, val_acc = model.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f' % val_loss)
    print('val_acc: %f' % val_acc)

    # ===============================
    # Prediction
    # ===============================
    x = x_test[:10]
    t = t_test[:10]
    y = softmax(model.predict(x))

    for i in range(len(y)):
        print('---------- prediction %d ----------' % (i+1))
        class_pred = np.argmax(y[i])
        prob_pred = y[i][class_pred]
        class_true = np.argmax(t[i])
        print('prediction:')
        print('\tclass: %d, probability: %f' % (class_pred, prob_pred))
        print('\tclass (true): %d' % class_true)

    # ===============================
    # Save model
    # ===============================
    print('saving model...')
    model.save('model.h5')
    del model

    # ===============================
    # Load model
    # ===============================
    print('loading model...')
    model = load_model('model.h5')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
```

## Notes

* In ELM, you can apply an activation function only to the hidden nodes.
* ELM always finds the global optimal solution for the weight matrices at every training.
* ELM does not need to train iteratively on the same data samples,
while backpropagation-based models usually need to do that.
* ELM does not update 'alpha', the weight matrix connecting the input nodes
and the hidden nodes. It reduces the computational cost by half.
* ELM does not need to compute gradients. The weight matrices are trained by
computing a pseudoinverse.
* The computational complexity for the matrix inversion is about O(batch\_size^3 \* n\_hidden\_nodes),
so take care for the cost when you increase batch\_size.

## Demo

You can execute the above sample code with the following command.

`$ python train.py [--n_hidden_nodes] [--activation] [--loss]`

* `--n_hidden_nodes`: (optional) the number of hidden nodes (default: 1024).
* `--activation`: (optional) an activation function applied to the hidden nodes.
we currently support `sigmoid` and `identity` (default: sigmoid).
* `--loss`: (optional) a loss function applied to the output nodes.
we currently support `mean_squared_error` and 'mean_absolute_error' (default: mean_squared_error).

ex) `$ python train.py --n_hidden_nodes 2048 --activation sigmoid --loss mean_absolute_error`

## Todos

* support more activation functions
* support more loss functions
* provide benchmark results
