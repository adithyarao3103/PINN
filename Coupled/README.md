# PINN for coupled differential equations

Here, we use PINNs to solve the coupled differential equations of the form 

$$
f' = \mathcal{U}(f,g)
$$

$$
g' = \mathcal{V}(f,g)
$$

In the case of the coupled DEs, we provide two possible implementations in PINNs. 

## 1. Using the two neural networks

[Jupyter notebook](PINN_coupledDe.ipynb)

In this case, we use two neural networks to solve the coupled DEs. The first neural network is used to represent $f$, and the second neural network is used to represent $g$. The two neural networks are trained simultaneously, the loss being the sum of the residues of the two DEs.

## 2. Using a single neural network

[Jupyter notebook](PINN_coupled_singleNN.ipynb)

In this case, we use a single neural network with two outputs, the first one giving the value of $f$, and the second one giving the value of $g$. The neural network is again trained to minimize the sum of the residues of the two DEs.

> Efficiency consideration: So far I have no addressed the question of which of the above two methods would be more efficient.