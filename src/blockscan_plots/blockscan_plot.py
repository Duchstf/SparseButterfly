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
    for i in range(1, 21, 2):
        loss1 = np.load(f'CIFAR10_MLP_Monarch{i}.npy')
        #loss2 = np.load('CIFAR10_MLP_Vanilla.npy')
        
        # Plot loss
        plt.plot(moving_average(loss1, 50), label=f'Monarch {i}', alpha = 0.3)
        #plt.plot(loss2, label='Vanilla')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig("loss_CIFAR10_blockscan.png")
    
if __name__ == '__main__':
    plot_loss()