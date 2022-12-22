# Diffusion model

<p align="center">
<img src="img/Diffusion_exemple.jpeg" alt="diffusion_example" style="height: 175px; width:500px;"/>
</p>

Diffusion models are the new state-of-art generative models that are used for image synthesis. They are called Denoise Diffusion Probabilistic Models (DDPMs) and are considered as score-based generative models.
The main goal of these models is to generate/synthetize image from noise by reversing the process. We can define diffusion model as a parameterized Markov chain trained using
variational inference to produce samples matching the data after finite time.[[1]](/papers/Denoising%20Diffusion%20Probabilistic%20Models.pdf)

The learning process is composed of a forward stochastic differential equation (SDE) that consist in converting data(images) slowly into noise. Then we entend to reverse the diffusion process yields score-based generative models to found back the original data by starting with the noise. This is the the backward propagation part
## Detailled process
###  Stochastic differential equation


In a first time, we need to specify an SDE that will the data distribution $p_0$ to a prior distribution  $p_T$. In a second time, we need to define the loss function. Now we just need to implement the forward propagation SDE and the reverse-time SDE. To solve the reverse-time SDE the Euler-Maruyama method is usually used.
[[2]](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)

## Our Diffusion model

We will use a U-net architecture for our DDPM. The U-net is a specifical Neural Network  in U shape.(e.g)
<p align="center">
<img src="img/u-net-architecture.png" alt="u-net" style="background-color:white;height: 200px; width:'400px;"/>
</p>

The U-net will first "compress" the image and in a second will blow the image to it's original size. The particularity is that we skip some connection from the previous layer to laters ones in order to add spatial and context information to the final output.

First we add noises progressively to the original image allowing the model to learn from it.
It's the forward process of our U-Net. We use the MNIST-fashion dataset to test our model.


<p align="center">
<img src="img/noise_adding0.png"  style="background-color:white;height: 150px; width:'200px;"/>
</p>


<p align="center">
<img src="img/noise_adding.png"  style="background-color:white;height: 150px; width:'200px;"/>
</p>

<p align="center">
<img src="img/noise_adding5.png"  style="background-color:white;height: 150px; width:'200px;"/>
</p>

<p align="center">
<img src="img/noise_adding75.png"  style="background-color:white;height: 150px; width:'200px;"/>
</p>
<p align="center">
<img src="img/noise_adding100.png"  style="background-color:white;height:150px; width:'200px;"/>
</p>

At each timesteps our model will optimise the MSE between the estimate noise from our model $\epsilon_\theta$ and random noise $\epsilon$ .
The final equation to minimise become:

<br />

$$
||ϵ - ϵ_\theta(x_t,t)||^2=||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)||^2
$$

<br />

# Result


<p align="center">
<img src="img/fashion.gif"  style="background-color:white;height: 300px; width:'400px;"/>
</p>














