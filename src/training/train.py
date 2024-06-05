import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.display import display
import csv
import pandas as pd 
import matplotlib.dates as md
from matplotlib.ticker import AutoMinorLocator

from gcns import GCNS1,GCNS2,GCNS3,GCNS4, GCNBlock
from dbgcn_utils import generate_dataset, load_data, get_normalized_adj

use_gpu = True
num_timesteps_input = 1
num_timesteps_output = 1

epochs = 40 # Number of times a training dataset pass through the algorithm
batch_size = 54 # Number of dataset to be passed in batches

parser = argparse.ArgumentParser(description='DBGCN')
parser.add_argument('--enable_cuda', action='store_true', help='Enable CUDA')
parser.add_argument('--data',type=str,default=r"C:\FILES\thesis-vsc-files\data\hk\hk_speed.npy",help='data path')
parser.add_argument('--adjdata',type=str,default=r"C:\FILES\thesis-vsc-files\data\hk\hk_adj.npy",help='adj data path')
parser.add_argument('--adj_lea', action='store_true', help='using the learned adj_matrix')
parser.add_argument('--save', type=str, default=r"C:\FILES\thesis-vsc-files\data\saved-model\saved",help='model save')
args = parser.parse_args()
args.device = None
args.mat_flag = False

if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

if args.adj_lea:
    args.mat_flag = True
else:
    args.mat_flag = False


def train_epoch(training_input, training_target, batch_size): # algorithm to train a dataset
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        if(i%50==0):
            print("Iter {:03d}: train loss is {:.4f}".format(i,epoch_training_losses[-1]))
    return sum(epoch_training_losses)/len(epoch_training_losses) # Loss function aggregated as a mean


if __name__ == '__main__':
    torch.manual_seed(7)

    A, X = load_data(args.data, args.adjdata) # Loading dataset

    Xshape = X.shape[2]
    split_line1 = int(Xshape * 0.6) # dataset partitioning
    split_line2 = int(Xshape * 0.8)

    print(X.shape, Xshape, split_line1, split_line2)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output) # Calling generate_dataset for training, validating, and testing dataset
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)
    
    print("*********** Data load successfully! *********")

    A_wave = get_normalized_adj(A, args.mat_flag) # Get the normalized adjacency matrix
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=args.device)
    
    ## Try using a different GCN structure, GCNS1 or GCNS2 or GCNS3 or GCNS4
    net = GCNS1(A_wave.shape[0],
               training_input.shape[3],
               num_timesteps_input,
               num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) # Using ADAM optimizer
    loss_criterion = nn.MSELoss() # Mean square error loss

    training_losses = []
    validation_losses = []
    validation_maes = []
    validation_mapes = []
    validation_rmses = []
    for epoch in range(epochs):
        ## training
        loss = train_epoch(training_input, training_target, batch_size=batch_size)
        training_losses.append(loss)
        ## validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)
            out = net(A_wave, val_input) # Defined GCN (neural network to be trained)
            val_loss = loss_criterion(out, val_target).to(device="cpu")
            validation_losses.append(np.ndarray.item(val_loss.detach().numpy()))

            out_unnormalized = out.detach().cpu().numpy()
            target_unnormalized = val_target.detach().cpu().numpy()

            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            mape = np.mean(np.absolute(out_unnormalized - target_unnormalized)/target_unnormalized)
            rmse = np.sqrt(np.mean((out_unnormalized - target_unnormalized)**2))
            validation_maes.append(mae)
            validation_mapes.append(mape)
            validation_rmses.append(rmse)

            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        print("Epoch {:03d}--Training loss: {:.4f}".format(epoch+1, training_losses[-1]))
        print("Epoch {:03d}--Validation loss: {:.4f}".format(epoch+1, validation_losses[-1]))
        print("Epoch {:03d}--Validation MAE: {:.4f}--Validation MAPE: {:.4f}--Validation RMSE: {:.4f}".format(epoch+1, validation_maes[-1], validation_mapes[-1], validation_rmses[-1]))
        # torch.save(net.state_dict(),args.save + "_epoch_" + str(epoch) + ".pth")

    ## plot train & validation losses
    plt.plot(training_losses, label="Training loss")
    plt.plot(validation_losses, label="Validation loss")
    ax = plt.gca()
    plt.xlabel("Epoch", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.rcParams.update({'font.size': 22}) 
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()
    plt.show()

    ## testing
    print("*************** Testing .... *****************")

    with torch.no_grad():
        net.eval()
        test_input = test_input.to(device=args.device)
        test_target = test_target.to(device=args.device)
        out = net(A_wave, test_input) # GCN (neural network) used
        test_loss = loss_criterion(out, test_target).to(device="cpu")
        test_loss = np.ndarray.item(test_loss.detach().numpy())

        out_unnormalized = out.detach().cpu().numpy()
        target_unnormalized = test_target.detach().cpu().numpy()

        test_mae = np.mean(np.absolute(out_unnormalized - target_unnormalized)) # mean absolute error
        test_mape = np.mean(np.absolute(out_unnormalized - target_unnormalized) / target_unnormalized) # mean absolute percentage error
        test_rmse = np.sqrt(np.mean((out_unnormalized - target_unnormalized) ** 2)) # root mean squared error

        out = None
        test_input = val_input.to(device="cpu")
        test_target = val_target.to(device="cpu")

    ground_shape = target_unnormalized.shape
    pred_shape = out_unnormalized.shape
    
    print("Testing finished")
    print("The test loss on best model is", str(round(test_loss,4)))
    print("--Testing MAE: {:.4f} --Testing MAPE: {:.4f} --Testing RMSE: {:.4f}".format(test_mae, test_mape, test_rmse))

    print("Groundtruth size:",ground_shape)
    print("Predictions size:",pred_shape)

ground = 70*target_unnormalized
pred = 70*out_unnormalized

# roadnets
roadnet1 = [0,1,39,167,76]
roadnet2 = [12,13,72,15,16]
roadnet3 = [3, 4, 9, 7]
for i in roadnet3:
    plt.plot(ground[:,i,0], label="Groundtruth")
    plt.plot(pred[:,i,0], label="Predictions")
    plt.legend()
    plt.ylim(0,100)
    plt.xlim()
    plt.xlabel("Time", fontsize=22)
    plt.ylabel("Speed (kph)", fontsize=22) 
    ax = plt.gca()
    plt.rcParams.update({'font.size': 22})
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.show()

# csv-file for heatmap
def array_to_csv(array, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(array)

#save-csv
array_to_csv(target_unnormalized,  r"C:\FILES\thesis-vsc-files\data\hk\roadnet_groundtruth.csv")
array_to_csv(out_unnormalized, r"C:\FILES\thesis-vsc-files\data\hk\roadnet_prediction.csv")

# Read the CSV file
df = pd.read_csv(r"C:\FILES\thesis-vsc-files\data\hk\roadnet_prediction.csv", header=None)
df1 = pd.read_csv(r"C:\FILES\thesis-vsc-files\data\hk\roadnet_groundtruth.csv", header=None)

# Function to remove square brackets
def remove_brackets(cell):
    if isinstance(cell, str):  # Ensure the cell is a string
        return cell.replace('[', '').replace(']', '')
    return cell

# Apply the function to all cells in the DataFrame
df = df.applymap(remove_brackets)
df1 = df1.applymap(remove_brackets)
# Save the cleaned DataFrame back to a CSV file
df.to_csv(r"C:\FILES\thesis-vsc-files\data\hk\roadnet_prediction.csv", index=False)
df1.to_csv(r"C:\FILES\thesis-vsc-files\data\hk\roadnet_groundnet.csv", index=False)