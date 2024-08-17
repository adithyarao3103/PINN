# Physics Informed Neural Networks (PINNs)

This repository contains Jupyter notebooks for the TensorFlow implementation of PINNs for solving PDEs and performing integrations (in a given domain, outside the domain the neural network fails to encapsulate the PDE).

I haven't trained them to very good precision. One can see that choosing a proper architecture and training routine can lead to better convergence, the presented notebooks are only to illustrate how PINNs are used

<!-- The 2D case, where I am trying to solve the wave equation is, for some reason, getting glitchy. While for the presented case, the solution is right, for other cases the solution does not quite match the expected solution. I need to look into the implementation to find out what is wrong with the implementation! -->
---

## PINNs

The primary idea behind PINNs is to use a neural network to approximate the solution of a PDE. This is based on the [Kolmogoroov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem), which states that any continuous multivariate function can be represented as a finite composition of continuous functions of single variables, and the binary operation of addition. That is, given a function $f(x_1, x_2, ..., x_n)$, we can write

$$
f(x_1, x_2, ..., x_n) = \sum_{i=1}^{2n+1} \Phi_q\left( \sum_{p=1}^{n} \phi_{q,p}(x_p)\right)
$$

We use the above theorem to represent the solution of a PDE as a neural network, and then train the neural network to minimize the residual of the PDE, i.e., the difference between the left-hand side and the right-hand side of the PDE. Therefore, given a PDE of the form 

$$
f(D_{x_1},~ \dots,~D_{x_n})[u(x_1, \dots, x_n)] = g(x_1, \dots, x_n)
$$

where $D_{x_i}$ is the partial derivative of $u$ with respect to $x_i$, we consider a neural network $u_\theta (x_1, ~\dots,~x_n)$ with parameters $\theta$ and train the neural network to minimize the loss 

$$
    \mathbb{E}\left(~f(D_{x_1},~ \dots,~D_{x_n}) [u_\theta(x_1, \dots, x_n)] - g(x_1, \dots, x_n)~\right) + \mathrm{Boundary~Conditions}
$$
where the $\mathbb{E}$ denotes the expectation value and is taken over a set of training points in the domain of the function $u$.

## Tensorflow implementation 

The implementation is based on the [TensorFlow](https://www.tensorflow.org/) library. The code is written in Python 3.11.0

In the code, we define the neural network class and use the `tf.GradientTape()` to compute the derivatives of the neural network with respect to the input variables. We then use the computed derivatives to obtain the residual of the PDE, and then the loss. 

Further, we use the Adam optimizer to minimize the loss. We see that even with small networks and a small number of training batches and epochs, we can obtain a very good approximation to the solution of the PDE.