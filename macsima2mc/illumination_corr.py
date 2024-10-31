from basicpy import BaSiC
import numpy as np

def indices_per_channel(total_imgs,no_of_channels):
    #total_imgs in the stack
    img_indices=[ list( range(ch,total_imgs,no_of_channels) ) for ch in range( 0, no_of_channels )   ]
    return img_indices

def extract_channel_imgs (stack,indices):
    return stack[indices,:,:]

def apply_corr(uncorr_stack,no_of_channels):
    corr_stack=np.zeros( uncorr_stack.shape,dtype=uncorr_stack.dtype )
    total_imgs=uncorr_stack.shape[0]
    indices=indices_per_channel(total_imgs, no_of_channels)
    basic = BaSiC(get_darkfield=False, smoothness_flatfield=1.0,fitting_mode = "approximate", sort_intensity = True) 
    for ind_list in indices:
        uncorr_imgs=extract_channel_imgs(uncorr_stack, ind_list)
        basic.fit(uncorr_imgs)
        ffp=basic.flatfield
        corr_stack[ind_list,:,:]=np.uint16( np.clip( uncorr_imgs.astype(float)/ffp  ,0, 65535) )
    return corr_stack


