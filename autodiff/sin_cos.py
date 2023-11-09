'''
This example will compute dy/dx for y=sin(x) using
automatic differentiation. It then plots y(x) and dy/dx on the same axes
'''
import autodiff
from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(-np.pi*2,np.pi*2,1000)
f = lambda x: autodiff.sin(x)

y = []
dy = []
for xi in autodiff.to_var(x):
    y_i = f(xi).value
    dy_i = autodiff.grad(f(xi))[xi]
    y.append(y_i)
    dy.append(dy_i)

plt.style.use("ggplot")
fig, ax = plt.subplots(figsize=(18,10))
ax.set(title=r"$y=sin(x)$ and $\frac{dy}{dx}$ computed with autodiff")
ax.plot(x,y, label=r"$y$")
ax.plot(x,dy, label=r"$\frac{dy}{dx}$")
ax.legend()
plt.show()

