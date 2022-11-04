# Diffusion model

<p align="center">
<img src="Diffusion_exemple.jpeg" alt="diffusion_example" style="height: 175px; width:500px;"/>
</p>

Diffusion models are the new state-of-art generative models that are used for image synthesis. They are called Denoise Diffusion Probabilistic Models (DDPMs) and are considered as score-based generative models.
The main goal of these models is to generate/synthetize image from noise by reversing the process.

The learning process is composed of a forward stochastic differential equation (SDE) that consist in converting data(images) slowly into noise. Then we entend to reverse the diffusion process yields score-based generative models to found back the original data by starting with the noise.
## Detail process
###  Stochastic differential equation


In a first time, we need to specify an SDE that will the data distribution $p_0$ to a prior distribution  $p_T$











