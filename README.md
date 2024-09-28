# Physics Informed Neural Networks (PINNs)

This repository contains Jupyter notebooks for the TensorFlow implementation of PINNs for solving Differential Equations (in a given domain, outside the domain the neural network fails to encapsulate the PDE).

(There is another application, which is to find the parameters of a DE by fitting a PINN to the “data”, which is the solution of the DE. This is not considered here. Rather we have, so far, only considered PINNs to find the solution for a given DE)

---

## PINNs

The primary idea behind PINNs is to use a neural network to approximate the solution of a PDE. This is based on the [Kolmogorov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem), which states that any continuous multivariate function can be represented as a finite composition of continuous functions of single variables, and the binary operation of addition. That is, given a function $f(x_1, x_2, ..., x_n): [0,1]^n \to \mathbb{R}$, we can write

$$
f(x_1, x_2, ..., x_n) = \sum_{i=1}^{2n+1} \Phi_q\left( \sum_{p=1}^{n} \phi_{q,p}(x_p)\right)
$$

where $\phi_{q,p} : [0,1] \to \mathbb{R}$ and $\Phi_q : \mathbb{R} \to \mathbb{R}$

We use the above theorem to represent the solution of a PDE as a neural network, and then train the neural network to minimize the residual of the PDE, i.e., the difference between the left-hand side and the right-hand side of the PDE. Therefore, given a PDE of the form 

$$
f(D_{x_1},~ \dots,~D_{x_n})[u(x_1, \dots, x_n)] = g(x_1, \dots, x_n)
$$

where $D_{x_i}$ is the partial derivative of $u$ with respect to $x_i$, we consider a neural network $u_\theta (x_1, ~\dots,~x_n)$ with parameters $\theta$ and train the neural network to minimize the loss 

$$
    \mathbb{E}\left(~f(D_{x_1},~ \dots,~D_{x_n}) [u_\theta(x_1, \dots, x_n)] - g(x_1, \dots, x_n)~\right) + \mathrm{Boundary~Conditions}
$$

where the $\mathbb{E}$ denotes the expectation value and is taken over a set of training points in the domain of the function $u$.

---



## Implementation 

We implement the PINNs using both the [TensorFlow](Tensorflow) and [PyTorch](PyTorch) libraries.

The project was initially implemented in TensorFlow, but later a PyTorch implementation of the same was also implemented.

---
> For more details, refer the readmes in the respective folders.