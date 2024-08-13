# Physics Informed Neural Networks (PINNs)

This repository contains Jupyter notebooks for the TensorFlow implementation of PINNs for solving PDEs and performing integrations (in a given domain, outside the domain the neural network fails to encapsulate the PDE).

The 1D case and the integration case are working well and fine (albeit I haven't trained them to very good precision. One can see that choosing a proper architecture and training routine can lead to better convergence, the presented notebooks are only to illustrate how PINNs are used)

The 2D case, where I am trying to solve the wave equation is, for some reason, getting glitchy. I need to look into the implementation to find out what is wrong with the implementation!
