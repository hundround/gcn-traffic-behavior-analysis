import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.display import display

from gcns import GCNS1,GCNS2,GCNS3,GCNS4
from dbgcn_utils import generate_dataset, load_data, get_normalized_adj

use_gpu = True
num_timesteps_input = 12
num_timesteps_output = 3

epochs = 40
batch_size = 32

parser = argparse.ArgumentParser(description='DBGCN')
parser.add_argument('--enable_cuda', action='store_true', help='Enable CUDA')

# dataset
parser.add_argument('--data',type=str,default=r"C:\FILES\thesis-vsc-files\data\hk\hk_speed.npy",help='data path')
parser.add_argument('--adjdata',type=str,default=r"C:\FILES\thesis-vsc-files\data\hk\hk_adj.npy",help='adj data path')

args = parser.parse_args()
args.device = None

if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

def train_epoch(training_input, training_target, batch_size):
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
    return sum(epoch_training_losses)/len(epoch_training_losses)

if __name__ == '__main__':
    torch.manual_seed(7)

    A, X = load_data(args.data, args.adjdata)

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)
    print("*********** Data load successfully! *********")

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=args.device)
    
    ## The model performance under GCNS1 structure is the best
    net = GCNS1(A_wave.shape[0],
               test_input.shape[3],
               num_timesteps_input,
               num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    validation_mapes = []
    validation_rmses = []

    for epoch in range(epochs):
        print("############### testing  ##############")
        loss = train_epoch(test_input, test_target, batch_size=batch_size)
        training_losses.append(loss)
        print("############### validation  ##############")
        # Run validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)
            out = net(A_wave, val_input)
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

        print("Epoch {:03d}--Training loss: {:.4f}".format(epoch, training_losses[-1]))
        print("Epoch {:03d}--Validation loss: {:.4f}".format(epoch, validation_losses[-1]))
        print("Epoch {:03d}--Validation MAE: {:.4f}--Validation MAPE: {:.4f}--Validation RMSE: {:.4f}".format(epoch, validation_maes[-1], validation_mapes[-1], validation_rmses[-1]))
    plt.plot(training_losses, label="Testing loss")
    plt.plot(validation_losses, label="Validation loss")
    ax = plt.gca()

    plt.xlabel("Epoch", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.rcParams.update({'font.size': 22})
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()
    plt.show()