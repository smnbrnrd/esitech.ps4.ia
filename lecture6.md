class: middle, center, title-slide
name: lecture6

# Machine Learning
## Lecture 6: Machine learning for physics
<br><br>
Simon BERNARD<br>
[simon.bernard@univ-rouen.fr](mailto:simon.bernard@univ-rouen.fr)<br><br>
.center.height-4em[![URN logo](assets/logo-urn-color.png)]


---
class: middle, center
# AI for science

---
# AI for Science

.row[
.col-55[
**Major successes of AI: digital applications**
- Images, videos, texts, video games, etc.
- Massively available data

**What about sciences? Can we use AI:**
- as an alternative to traditional approaches?
- to solve problems more effectively?
- to accelerate scientific discovery?
]
.col-45[
.center.width-100[![](./medias/lec6/ai4science_nature.png)*[Wang et al, "Scientific discovery in the age of artificial intelligence", Nature, 2023](https://www.nature.com/articles/s41586-023-06221-2)*]
]
]

---
# AI for Science: Notable recent works

## Weather forecasting

.row[
.col-55[
- **GraphCast** (2023) .exponent[(1)]
- Training data: 40+ years of observations (ECMWF)
- Prediction of weather fields up to 10 days ahead
- 1.2 billion param., 128 TPUv4 chips for training

.center.width-80.mt-2[![](./medias/lec6/graphcast-result.gif)]
]
.col-45[
.center.width-90.mt-2[![](./medias/lec6/graphcast-model.png)*[https://deepmind.google/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/](https://deepmind.google/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/)*]
]
]

.footnote[(1) Renamed with newer versions: GenCast in 2024 and WeatherNext in 2025 ([github](https://github.com/google-deepmind/weathernext))]

---
# AI for Science: Notable recent works

## Nuclear fusion for sustainable energy production

- High-performance, clean and virtually inexhaustible energy
- Very difficult to control due to the unstable state of the plasma used for fusion
- AI for modeling and controlling the plasma:
  - Based on reinforcement learning
  - Simulated data and control policy that require significant domain expertise

.row[
.col-50[
.center.width-90.mt-2[![](./medias/lec6/nuclear-fusion-model.webp)]
]
.col-50[
.center.width-90.mt-2[![](./medias/lec6/nuclear-fusion-simulation.gif)]
]
]
.figcaption[**[Degrave et al., "Magnetic control of tokamak plasmas through deep reinforcement learning", Nature, 2023](https://www.nature.com/articles/s41586-021-04301-9)**]


---
# AI for Science: Notable recent works

## Protein structure prediction

.row[
.col-65[
- **Alphafold 2 (2021) and Alphafold 3 (2023)**
- Trained on 200k+ known structures
- Predictions for almost all known proteins (∼ 200M)
- Inspired by LLMs and with an iterative generation approach

.center.width-100[<video src="./medias/lec6/alphafold.mp4" controls autoplay loop muted playsinline></video>]
]
.col-35[
.center.width-90[![](./medias/lec6/alphafold-nature.png)]
]
]

---
# Scientific Machine Learning

.row[
.col-50[
**Limitations:**
- Based on architectural priors
- Need for a large amount of data
- No interpretability and trustworthiness
- Poor generalization to new regimes 

**Contradictory to the scientific approach:**
- Hypothesis, theory and experiments
- Must be interpretable and trustworthy
- Generalization to new regimes is the goal
- Must be consistent with scientific knowledge
]
.col-50[
**Solution:**
.center.width-80.mt-2[![](./medias/lec6/sciml.png)*S. Mishra, B. Mosley, ETH Zürich, ["AI in the Sciences and Engineering"](https://camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html) (2024)*]
]
]

---
# Scientific Machine Learning
## Three main ways to incorporate scientific constraints

.row.evenly.stretch.center.mt-5[
.col-30.border[
**Soft constraint**

Add a term to the loss function that penalizes violations of scientific laws (e.g. PDE residuals)

.width-80[![](./medias/lec6/sciml_approach1.png)]
]
.col-30.border[
**Hard constraint**

Adapt the architecture (e.g. re-parametrizing the output) to ensure compliance with a scientific law

.width-80[![](./medias/lec6/sciml_approach2.png)]
]
.col-30.border[
**Hybrid**

Integrate a model into a scientific process, such that ML is not a surrogate, but a building block

.width-98[![](./medias/lec6/sciml_approach3.png)]
]
]

---
# What are the machine learning tasks?
## Partial Differential Equations are building blocks of science

- Partial derivative of $u(x\_1, \ldots, x\_n) : \mathbb{R}^n \to \mathbb{R}$:
$$
  \frac{\partial u}{\partial x\_i} = \lim\_{\Delta x\_i \to 0} \frac{u(x\_1, \ldots, x\_i + \Delta x\_i, \ldots, x\_n) - u(x\_1, \ldots, x\_i, \ldots, x\_n)}{\Delta x\_i}
$$
i.e. the magnitude of variation of $u$ along the $x\_i$ direction
- Partial differential equations (PDEs):
$$
  F\left(x\_1, \ldots, x\_n, u, \frac{\partial u}{\partial x\_1}, \ldots, \frac{\partial u}{\partial x\_n}, \frac{\partial^2 u}{\partial x\_1^2}, \frac{\partial^2 u}{\partial x\_1 \partial x\_2}, \ldots \right) = 0
$$
- Exemple: the heat equation
$$
  \frac{\partial u}{\partial t} - \alpha \nabla^2 u = 0, \quad u: \mathbb{R}^d \times [0, T] \to \mathbb{R}
$$
where $u$ is the temperature field, $\alpha$ is the thermal diffusivity, and $\nabla^2$ is the Laplacian operator

---
count: false
# What are the machine learning tasks?
## Partial Differential Equations are building blocks of science

- Understanding and solving these equations is crucial for many scientific and industrial applications: aerodynamic simulation, weather forecasting, biological modeling, etc.
- **Numerical solvers are time-consuming, computationally intensive, and sometimes complex**

.row[
.col-50[
.center.width-90[![](./medias/lec6/dynamicalsystem_examples.png)*Yuan YIN, soutenance de thèse, Sorbonne Université, 2023*]
]
.col-50[
.center.width-60[![](./medias/lec6/Flow_around_a_wing.gif)]
.center.width-80[![](./medias/lec6/Cylinder_flow.gif)]
]
]

---
# What are the machine learning tasks?

Generic form of nonlinear PDEs:
$$
\begin{cases}
  \mathcal{N}[u, \mathbf{x}, t; \mu] = 0, &\quad \mathbf{x} \in \Omega \\\\
  u(\mathbf{x},t) = g(\mathbf{x},t), &\quad \mathbf{x} \in \partial\Omega \text{ (boundary condition)} \\\\
  u(\mathbf{x},0) = u\_0(\mathbf{x}), &\quad \mathbf{x} \in \Omega \text{ (initial condition)}
\end{cases}
$$
where $u$ is the unknown field, $\mu$ are the PDE parameters, $\mathcal{N}$ is a differential operator

The three main tasks in scientific machine learning are:
- **Simulation**: solving a PDE for given initial and boundary conditions, i.e. finding $u$ at any $(\mathbf{x}, t)$
- **Inverse problems**: inferring unknown parameters of a PDE from measurements, i.e. finding $\mu$
- **Scientific discovery**: finding new scientific laws from data, i.e. finding $\mathcal{N}$


---
# Example: Navier-Stokes equations for incompressible flow

$$
    \begin{cases}
        \dfrac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f} & \text{(momentum equation)} \\\\
        \nabla \cdot \mathbf{u} = 0 & \text{(incompressibility condition)}
    \end{cases}
$$
with $\mathbf{u}$ the velocity field, $p$ the pressure, $\nu$ the kinematic viscosity and $\mathbf{f}$ the external forces

.center.width-70.mt-2[![](./medias/lec6/navier-stokes-initialconditions.gif)*[https://zongyi-li.github.io/neural-operator/](https://zongyi-li.github.io/neural-operator/)*]


---
class: middle, center

# Physics-Informed Neural Network

---
# Implicit Neural Representation (or Coordinate-based Network)

- So far, a NN was a **predictor**: encode a *rule* extracted from a dataset to map a sample to a label
- INR: the network maps **coordinates** to the **value of a field** at those coordinates
$$h\_{\theta}: \mathbf{x} \longmapsto f(\mathbf{x})$$
- The NN no longer encode a rule learned from many examples, it encodes **one single object**.

.row[
.col-50.center[
.width-80[![](./medias/lec6/inr_network.png)*The network takes coordinates as input and outputs the value of the field at those coordinates*]
]
.col-50.center[
  .width-75[![](./medias/lec6/inr_image.png)]
]
]

---
class: middle, dark-slide

# Notebook [lec6-inr.ipynb](./notebooks/lec6-inr.ipynb)

---
count: false
# Implicit Neural Representation (or Coordinate-based Network)

Three consequences, and they are the whole point:

.box[**Continuous** — the field is defined everywhere, not only at the nodes of a grid. Query any $\mathbf{x}$, at any resolution.]

.box[**Compact** — the memory cost is the number of weights, no longer the number of pixels or mesh cells.]

.box[**Differentiable** — $\partial h\_{\theta} / \partial \mathbf{x}$ is available *exactly*, by automatic differentiation.]

.important.mt-3[This last property is the one that changes everything.]

---
# From representing a field to solving an equation

.row[
.col-50[
- $u\_{\theta}(\mathbf{x}, t)$ is differentiable: $\partial\_{t} u\_{\theta}$ and $\partial\_{xx} u\_{\theta}$ are obtained exactly, by autodiff
- We can **evaluate the residual** $\mathcal{N}[u\_{\theta}]$ at any point of the domain... *without knowing its solution*
- Core idea: instead of fitting the network to known **values**, fit it to the **equation** itself
]
.col-45.center[
.width-95[![](./medias/lec6/autodiffresidual.png)]
]
]

.important[Supervised learning minimizes a distance to **labels**. A PINN minimizes a distance to an **equation**.]

---
# Physics-Informed Neural Network (PINN) .exponent[(1)]

.center.width-95[![](./medias/lec6/pinnarchitecture.png)]

.footnote[(1) Maziar Raissi, Paris Perdikaris, George Em Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations", Journal of Computational Physics, 2019]

---
# The residual loss

Given a PDE written as $\mathcal{N}[u] = 0$ on a domain $\Omega$, with boundary and initial conditions:

$$\mathcal{L}(\theta) = \lambda\_{r} \mathcal{L}\_{r}(\theta) + \lambda\_{b} \mathcal{L}\_{b}(\theta) + \lambda\_{d} \mathcal{L}\_{d}(\theta)$$

.row.mt-1.small[
.col-32[
**PDE residual**
$$\mathcal{L}\_{r} = \frac{1}{N\_{r}} \sum\_{i=1}^{N\_{r}} \big| \mathcal{N}[u\_{\theta}(\mathbf{x}\_{i}, t\_i)] \big|^{2}$$
]
.col-02[]
.col-32[
**Boundary / initial**
$$\mathcal{L}\_{b} = \frac{1}{N\_{b}} \sum\_{j=1}^{N\_{b}} \big| u\_{\theta}(\mathbf{x}\_{j}, t\_j) - g\_{j} \big|^{2}$$
]
.col-02[]
.col-32[
**Measurements**
$$\mathcal{L}\_{d} = \frac{1}{N\_{d}} \sum\_{k=1}^{N\_{d}} \big| u\_{\theta}(\mathbf{x}\_{k}, t\_k) - u\_{k} \big|^{2}$$
]
]

- The $\mathbf{x}\_{i}$ are **collocation points**, sampled anywhere in $\Omega$ — no mesh, and **no labels needed**
- The third term is **optional**: a PDE can be solved with zero measurements

---
# Example: the Burgers' equation

Given $\nu = 0.01 / \pi$, $x \in [-1, 1]$, $t \in [0, 1]$ and $u: [-1, 1] \times [0, 1] \to \mathbb{R}$:

.row[
.col-40[
$$
\begin{cases}
  \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0\\\\
  u(x,0) = -\sin(\pi x)\\\\
  u(-1,t) = u(1,t) = 0
\end{cases}
$$

.small.mt-5[Ci-contre: INR trained with 
- $100$ points at $t=0$, $x=-1$ and $x=1$ 
- $10,000$ collocation points sampled in the domain $[-1, 1] \times [0, 1]$]
]
.col-55.center[
.center.width-100.mt-2[![](./medias/lec6/pinnburgers.png)]
]
]

---
class: middle, dark-slide

# Notebook [lec6-pinn-oscillator.ipynb](./notebooks/lec6-pinn-oscillator.ipynb)

---
# What PINNs are good for — and where they stop

.box[**Inverse problems** — an unknown physical parameter $\mu$ becomes a *trainable variable*, optimized jointly with $\theta$. No outer loop of forward solves.]

.box[**Sparse and noisy measurements** — the PDE acts as a physical regularizer and fills the gaps: a full velocity field from a handful of sensors.]

.box[**Awkward geometries** — nothing to mesh, and the cost does not explode with the dimension of the domain.]

.alert.mt-3[**But one network solves one problem instance.** Change the initial condition, the boundary conditions or a coefficient, and everything has to be retrained from scratch.]

.important[A PINN is a **solution**. What we would really like is a **solver**.]

---
class: middle, center

# Neural Operators

---
# Learning the solution operator

.row[
.col-60[
- PINN = **one** triple $(\mu, g, u\_0)$. 
- Change the initial condition and it needs to be retrained
- Solver = **map between functions**
$$\mathcal{G}: u\_0 \longmapsto u$$
from the initial condition to the whole solution
- A **neural operator** learns $\mathcal{G}$ itself, once, from a set of instances already solved by a classical solver
]
.col-40.center[
.width-95[![](./medias/lec6/neuraloperator.png)]
]
]

---
# Why a convolutional network is not enough

.row[
.col-58[
- Natural idea: sample $u\_0$ on a $64 \times 64$ grid, feed it to a CNN, output $u$ on the same grid. **It works.**
- But what has been learned is a map between **vectors of fixed size**, not between functions
- Change the resolution, the mesh or the geometry, and the model is useless
]
.col-42.center[
.width-95[![](./medias/lec6/discretizationtrap.png)]
]
]

.box[What we want is **discretization invariance**: a model trained at one resolution that can be evaluated at another, because it has learned something about the *functions*, not about the *samples*.]

---
# Fourier Neural Operator: the idea .exponent[(1)]

.row[
.col-56[
- For a linear PDE, the solution is a convolution with the Green's function
$$u(\mathbf{x}) = \int G(\mathbf{x} - \mathbf{x}')\, f(\mathbf{x}')\, d\mathbf{x}'$$
- So the solution operator is, in essence, a **global convolution**
- And a global convolution is a **pointwise product in Fourier space**
- So: go to Fourier, multiply by learned coefficients, come back
]
.col-44.center[
.width-85[![](./medias/lec6/fnoidea.png)]
]
]

.box[Lecture 4: a CNN kernel is **local** ($3 \times 3$) and defined on the grid. A Fourier kernel is **global** and defined on *frequencies* — a property of the function, not of the sampling.]

.footnote[(1) Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021]

---
# Fourier Neural Operator: how it works

.center.width-95.mt-2[![](./medias/lec6/fno.png)]

$$v\_{t+1}(\mathbf{x}) = \sigma\Big( \underbrace{\mathbf{W} v\_{t}(\mathbf{x})}\_{\text{local}} + \underbrace{\mathcal{F}^{-1}\big( R\_{\phi} \cdot \mathcal{F}(v\_{t}) \big)(\mathbf{x})}\_{\text{global}} \Big)$$


---
# What it buys you

.row[
.col-48[
**Discretization invariance**

Train at $64 \times 64$, evaluate at $256 \times 256$ without retraining — the Fourier modes do not care about the grid

**Speed**

Three orders of magnitude faster than the solver, for a *whole family* of instances
]
.col-04[]
.col-48[
.center.width-100[![](./medias/lec6/fnoresolution.png)]
]
]

.row.mt-3[
.col-48[
.box[**PINN** — learns *a solution*. No data. One instance. Retrain for anything new.]
]
.col-04[]
.col-48[
.box[**Neural operator** — learns *a solver*. Needs solved examples. Any instance, instantly.]
]
]

---
class: middle, center

# Perspectives

---
# AI for Science: The hot topic

- Very active research topic in the AI community since &sim;2020
- Goal is to design new AI methods dedicated to scientific and engineering problems

.row[
.col-50[
.center.width-95[![](./medias/lec6/ai4science_papers.png)*Ben Blaiszik, [https://github.com/blaiszik/ml_publication_charts/](https://github.com/blaiszik/ml_publication_charts/)*]
]
.col-50[
.center.width-80[![](./medias/lec6/ai4science_conf.png)*S. Mishra, B. Mosley, ETH Zürich, ["AI in the Sciences and Engineering"](https://camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html) (2024)*]
]
]

---
# Foundation models for science

.row[
.col-55[
- **AlphaFold 2/3**: protein structure prediction, a 50-year-old problem essentially solved at CASP14
- **GraphCast**, **Pangu-Weather**, **Aurora**: medium-range weather forecasting, more accurate than the operational physical model, in seconds on one GPU instead of hours on a supercomputer
- **GNoME**, **MatterGen**: materials discovery at the scale of millions of candidate crystals
]
.col-45.center[
.width-95[![](./medias/lec6/foundationmodels.png)]
]
]

- Same recipe every time: a huge corpus (often *simulated*), strong architectural priors, one model reused across many downstream tasks
- Exactly the pretraining paradigm of lecture 5, applied to science

---
# Pitfalls and good practice

- **Data leakage**: splitting at random over samples coming from the same run gives absurdly optimistic scores. Split by **configuration**, by **experiment**, or by **time**
- **Validate on physics, not only on the MSE**: conservation of energy and mass, symmetries, asymptotic regimes, dimensional consistency
- **State the validity domain** of the model, and detect inputs that fall outside it
- **Report uncertainty**, always
- **Compare to honest baselines**: a linear model, a spline interpolation, or the classical solver *at equal computational budget*
- **Reproducibility**: version the data-generation scripts, not just the model weights

.center.width-60.mt-2[![](./medias/lec6/goodpractice.png)]

---
# Takeaways

.box[**Physics is not a generic ML problem**: little data, known laws, strong symmetries, and extrapolation as the actual goal.]

.box[**Surrogate models** trade accuracy outside the training envelope for a $10^{3}$–$10^{6}$ speed-up, and are differentiable.]

.box[**Physics can be injected in the data, in the loss, or in the architecture.** Soft constraints are flexible, hard constraints are exact.]

.box[**Inverse problems are where machine learning brings the most**, because the classical approach requires thousands of forward solves.]

.box[**No error bar, no physics.** Always quantify uncertainty and state the validity domain.]
