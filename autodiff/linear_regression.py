import autodiff
from matplotlib import pyplot as plt
plt.style.use("ggplot")
from autodiff import Var, to_var, to_vals
import numpy as np
import sys
sys.setrecursionlimit(10_000)

# load dataset
data_path = "./datasets/kleibers_law_data.csv"
data = np.loadtxt(data_path,delimiter=",")
x_vals = np.log(data[:-1,:])
y_vals = np.log(data[-1:,:])

# vectorize the input data to be ndarray<autodiff.Var>
x = to_var(x_vals)
y = to_var(y_vals)

# linear model
model = lambda x, w: (w[0] + np.dot(x.T, w[1:])).T

# loss function
def least_squares(w):
    cost = np.sum( (model(x,w) - y)**2 )
    return cost/float(y.size) # return average cost

def gradient_descent(g, alpha, max_iter, w):
    '''
    Minimizes the loss function g using gradient descent
    and returns the loss history and weigth history
    '''
    loss_hist = [g(w).value]
    weight_hist = [to_vals(w)]

    for k in range(max_iter):
        # evaluate gradient of loss function g at weights
        loss = g(w)
        grad_loss = autodiff.grad(loss)

        # update weights with gradient descent step
        w[0].value -= alpha * grad_loss[w[0]]
        w[1].value -= alpha * grad_loss[w[1]]
        
        # keep track of loss and weight history
        loss_hist.append(loss.value)
        weight_hist.append(to_vals(w))

    return loss_hist, weight_hist

# run gradient descent
w0 = to_var(np.array([10.0,10.0]))
loss_hist, weight_hist = gradient_descent(g=least_squares, alpha=0.02, max_iter=200, w=w0)

# plot the loss history
fig, ax = plt.subplots()
ax.plot(loss_hist)
ax.set(title="Loss history")

# Using the latest weight, plot a line with the learned slope
# along with the original data
w = weight_hist[-1] 
xfit = np.arange(-6,8,0.01)
yfit = [w[0] + xi*w[1] for xi in xfit]
fig, ax = plt.subplots(figsize=(10,7))
ax.scatter(x_vals[0],y_vals[0],color="blue")
ax.set(xlabel="log of mass",ylabel="log of metabolic rate")
ax.plot(xfit,yfit,"-r")
plt.show()
