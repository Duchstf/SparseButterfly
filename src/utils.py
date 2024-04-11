import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

# Load style sheet
plt.style.use(hep.style.CMS)  # or ATLAS/LHCb2

def plot_diag_weight(fc):
    '''
    Monarch matrices are block-diagonal matrices with diagonal blocks.
    This function plots the diagonal blocks of a Monarch matrix.
    
    (nblocks, block_size, block_size)
    '''
    print("Plotting layer's weight")
    
    w = fc.weight.detach().cpu().numpy()
    
    print("W Shape: ", w.shape)
    print("Saving factor: ", fc.saving)
    
    #Initialize the empty weight matrix
    #And then fill it in later
    monarch = np.zeros((fc.out_features_extended, fc.in_features_extended))
    print("Monarch Shape: ", (fc.out_features_extended, fc.in_features_extended))
    
    #Now set the values to the diagonal blocks
    nblocks = w.shape[0]
    out_blksz = w.shape[1]
    in_blksz = w.shape[2]
    
    for i in range(nblocks):        
        monarch[i * out_blksz : (i + 1) * out_blksz, i * in_blksz :(i + 1) * in_blksz] = w[i]
    
    plt.imshow(monarch, cmap='Reds', vmin=0., vmax=0.1)
    plt.show()
    plt.savefig("plots/mornarch_fc0.png")
