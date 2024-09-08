# Main file for nonlinear dynamical system example
import numpy as np
import scipy
import scipy.integrate
import scipy.io
import torch
import matplotlib.pyplot as plt
import os

# custom modules
import systems
from ResNetMZ import *

##################################################
# Hongli Zhao, honglizhaobob@uchicago.edu
# 
# Date: 08/19/2022
##################################################

# fix random seed
seed = 10
np.random.seed(seed)
torch.manual_seed(10)

## Generate data

# number of trajectories
nmc = 10000
# time grid size
dt = 0.02
t_end = 100
tgrid = np.arange(0, t_end, dt)
nt = len(tgrid)

# save data
mc_data = np.zeros([int(nmc), int(nt)])
filepath = "./data/nonlinear/"
filepath = filepath + "mc_data" + ".mat"
if os.path.isfile(filepath):
    # if MC data is already generated, load data
    data = scipy.io.loadmat(filepath)
    mc_data = data['MC Data']
else:
    # parameters
    alpha = 0.1
    beta = 8.91
    # run simulations
    for i in range(int(nmc)):
        print(i)
        # draw uniform random initial condition from [-2, 2] x [-4, 4]
        z0 = np.random.uniform(low=-2, high=2)
        z1 = np.random.uniform(low=-4, high=4)
        x0 = np.array([
            z0, 
            z1
        ])
        exact_path = scipy.integrate.odeint(systems.nonlinear_system, x0, tgrid, args=(alpha, beta, ))
        # only save data for the first variable
        mc_data[i, :] = exact_path[:, 0]
    # save Monte Carlo data
    scipy.io.savemat(filepath, {"MC Data": mc_data})

## Generate training data by slicing ground truth trajectory
# number of memory terms
n_mem = 20  # 3, 5, 8, 10, 13, 15, 18, 20
# memory length
T_mem = n_mem*dt
# number of re-samples along each trajectory
J0 = 5 
# take (n_mem+2) entries from the exact trajectory, used for training
X = np.zeros([nmc, J0, n_mem+1])
y = np.zeros([nmc, J0, 1])
# a short time span of training data should suffice
K = 50
t_train = K*dt
assert t_train > T_mem
# nuber of terms spanned
n_train = int(t_train/dt)+1
for i in range(nmc):
    for j in range(J0):
        # start at a random position of the training time series
        idx = np.random.randint(n_train-n_mem-2)
        X[i, j, :] = mc_data[i, idx:idx+n_mem+1]
        y[i, j, :] = mc_data[i, idx+n_mem+1]
X = np.reshape(X, [nmc*J0, n_mem+1])
y = np.reshape(y, [nmc*J0, 1])

plotit = True
if plotit:
    # visualize training data
    plt.figure(1, figsize=(15, 5))
    plt.plot(X[0:100, :].T)
    plt.scatter(np.repeat([n_mem+1], 100), y[0:100], s=0.5, color='black', label=r'$z_{n_{mem}+2}$')
    plt.title("Visualizing Training Data")
    plt.xlabel(r"$n_{mem}$")
    plt.ylabel(r"$x_1$")
    plt.legend()
    plt.show()

## Set up ResNet model
model = ResNet(n_mem)
model_path = "./data/nonlinear/trained_model"
model_loaded_flag = False
if os.path.isfile(model_path):
    # load trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model_loaded_flag = True

# loss function
loss_fn = torch.nn.MSELoss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
# create data loader
dataset = ReducedOrderDataset(X, y)
b = 500 # batch size
training_loader = torch.utils.data.DataLoader(dataset, batch_size=b, shuffle=True, drop_last=True)

# if not loading saved model, train one and save
if not model_loaded_flag:
    # main training loop
    num_epochs = 500
    # all loss values over epoch
    all_loss = []
    for epoch in range(num_epochs):
        running_loss = 0
        count_batch = 0
        for i, data in enumerate(training_loader):
            X_sample, y_sample = data
            # zero gradient
            optimizer.zero_grad(set_to_none=True)
            # make a prediction
            y_pred = model(X_sample)
            # compute loss
            loss = loss_fn(y_pred.float(), y_sample.reshape(-1, 1).float())
            # backprop
            loss.backward()
            optimizer.step()
            
            # count number of batches
            count_batch += 1
            running_loss += loss.item()
            
            # reporting
            if epoch % 10 == 0:
                print("... Epoch {}, Batch {}, loss = {}".format(epoch, count_batch, loss.item()))
        # record training loss
        all_loss.append(running_loss / count_batch) 
        # step scheduler
        scheduler.step()
# save trained model
model.save(model_path)

# make predictions
num_predictions = 5000-(n_mem+1)
all_predictions = model.predict(mc_data[:5, 0:n_mem+1], num_predictions)

## Plotting
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
for i in range(5):
    ax[0].plot(mc_data[i, :num_predictions+n_mem])
    ax[1].plot(all_predictions[i, :].detach().numpy())
ax[0].set_title(r"Ground Truth $x_1$")
ax[1].set_title(r"Predicted $x_1$")
ax[0].grid(True)
ax[1].grid(True)
plt.show()

## Compute and plot pointwise MSE
ground_truth = mc_data[:5, :(n_mem+1+num_predictions)]
data_driven_mz_prediction = all_predictions.detach().numpy()
# MSE error
mse = np.mean((ground_truth - data_driven_mz_prediction)**2, 0)
# plot log scale
plt.plot(np.log(mse), color='green', label='log MSE')
plt.xlabel("Time Step")
plt.ylabel("Mean Squared Prediction Error")
plt.legend()
plt.show()

