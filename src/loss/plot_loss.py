import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

def plot_loss():
    
    # Load style sheet
    plt.style.use(hep.style.CMS)
    
    # Load loss
    loss1 = np.load('CIFAR10_MLP_Monarch.npy')
    loss2 = np.load('CIFAR10_MLP_Vanilla.npy')
    
    # Plot loss
    plt.plot(loss1, label='Monarch')
    plt.plot(loss2, label='Vanilla')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("loss_CIFAR10.png")
    
if __name__ == '__main__':
    plot_loss()