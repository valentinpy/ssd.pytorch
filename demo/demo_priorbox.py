import torch
from models.vgg16_ssd import build_ssd
from layers import *
from data import kaist

if __name__ == '__main__':

    # prepare environnement
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    #build_ssd('test', 300, 2, 'KAIST')  # initialize net

    priorbox = PriorBox(kaist)
    with torch.no_grad():
        priors = priorbox.demo(mode="vert_only")
        # priors = priorbox.demo(mode="vert_med")
        # priors = priorbox.demo(mode="all")
