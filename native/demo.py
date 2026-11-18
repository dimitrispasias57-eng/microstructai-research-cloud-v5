"""First-party synthetic software fixture, not biological validation."""
import numpy as np
def synthetic_image():
    yy,xx=np.mgrid[:96,:96]
    return np.clip(155-80*np.exp(-((np.sqrt((xx-47.5)**2+(yy-47.5)**2)-18)/2)**2),0,255).astype('uint8')
