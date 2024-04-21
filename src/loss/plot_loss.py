import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def plot_loss():
    
    # Load style sheet
    plt.style.use(hep.style.CMS)
    
    # Load loss
    loss1 = np.load('CIFAR10_MLP_Monarch.npy')
    loss2 = np.load('CIFAR10_MLP_Vanilla.npy')
    loss1_test = np.load('CIFAR10_MLP_Vanilla_test.npy')
    loss2_test = np.load('CIFAR10_MLP_Monarch_test.npy')
    
    # Plot loss
    plt.plot(moving_average(loss1, 50), label='Butterfly + Low Rank (train)')
    plt.plot(moving_average(loss2, 50), label='Vanilla (train)')
 
    plt.scatter(np.linspace(0, len(loss2), len(loss2_test)), loss2_test, label='Butterfly + Low Rank(test)')
    plt.scatter(np.linspace(0, len(loss1), len(loss1_test)), loss1_test, label='Vanilla (test)')

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("loss_CIFAR10.png")
    
if __name__ == '__main__':
    plot_loss()