from skimage.data import brain
from skimage.transform import resize, rescale

import numpy as np
import device
import torch, torch.nn as nn
import cv2


# visualize the image in a window using cv2
def cv2disp(win, ima, xp, yp, sc):
    cv2.imshow(win, rescale(ima, sc, False) * 1.0/(np.max(ima)+1e-15))
    cv2.moveWindow(win, xp, yp)


def np_to_00torch(np_array):
    # converts 2D np array to 4D torch array and is very important!
    return torch.from_numpy(np_array).float().unsqueeze(0).unsqueeze(0) # converts to 4D array, 1D = batch, 2D = channels, 3/4D are spatial dimensions


# to visualize we have to convert torch back to numpy
def torch_to_np(torch_array):
    return np.squeeze(torch_array.detach().cpu().numpy())


nxd = 128   # number of samples in the x-dimension 128x128 pixels
disp_scale = 4
brain_image = brain()

# parallel lines at 45 degrees means root(2) is the largest if you wanna consider all pixel
nrd = int(nxd*1.42) # number of radio bins, the largest it can be

nphi = nxd   # number of viewing angles

device = torch.device("cudo:0" if torch.cuda.is_available() else "cpu")

# CT Slice and convert it to torch tensor
true_object_np = resize(brain_image[5,30:-1,:-30], (nxd, nxd), anti_aliasing=False)     # extract a specific "slice" as image
true_object_torch = np_to_00torch(true_object_np).to(device)    # converts a numpy to a torch

cv2disp("True", true_object_np,0,0, disp_scale)


# -------------------- Torch system matrix ------------------------
def make_torch_system_matrix(nxd, nrd, nphi):
    system_matrix = torch.zeros(nrd * nphi, nxd * nxd)  # rows = num sinogram bins,, columns = num image pixels
    for xv in range(nxd):
        for yv in range(nxd):   # now have selected pixel(xv, yv)
            for ph in range(nphi):  # now for each angle project that pixel
                yp = -(xv-(nxd*0.5)) * np.sin(ph * np.pi/nphi) + (yv-(nxd*0.5)) * np.cos(ph * np.pi/nphi)
                # -(x * sin() + y * cos()) FBP
                # substract halv the number to define a center/origin as the middle of the image
                # np.pi/nphi to go between 0 and pi

                yp_bin = int(yp + nrd/2.0)
                # nrd = number of radial samples
                system_matrix[yp_bin + ph*nrd, xv + yv*nxd] = 1.0
                # converting a 2D to 1D to build a system matrix, column vector
    return system_matrix


# forward projection
# takes the image
def forward_proj_system_torch(image, sys_mat, nxd, nrd, nphi):
    return torch.reshape(torch.mm(sys_mat, torch.reshape(image, (nxd*nxd,1))), (nphi, nrd)) # reshape into sino


# back projection
# takes the sinogram
def back_proj_system_torch(sino, sys_mat, nxd, nrd, nphi):
    return torch.reshape(torch.mm(sys_mat.T, torch.reshape(sino, (nrd*nphi,1))), (nxd,nxd))


# create a system matrix
print(device)

syst_mat = make_torch_system_matrix(nxd, nrd, nphi).to(device)

true_sinogram_torch = forward_proj_system_torch(true_object_torch, syst_mat, nxd, nrd, nphi)
cv2disp("Sinogram", torch_to_np(true_sinogram_torch), disp_scale*nxd, 0, disp_scale)


class MLEM_Net(nn.Module):  # torch.nn is the base class for all pytorch neural network modules

    # this is the setup like for the iterative MLEM
    def __init__(self, sino_for_reonstruction, num_iter):
        super(MLEM_Net, self).__init__()    # inherit attributes and methods from base class, torch.nn standard requirement!
        self.num_its = num_iter # number of iterations
        self.sino_ones = torch.ones_like(sino_for_reonstruction) # sinogram just filled with ones
        self.sens_image = back_proj_system_torch(self.sino_ones, syst_mat, nxd, nrd, nphi) # sensitivity image = (A^T 1)

    def forward(self, sino_for_reconstruction):
        recon = torch.ones(nxd, nxd).to(device) # initial image filled with ones in x and y
        for it in range(self.num_its):
            fpsino = forward_proj_system_torch(recon, syst_mat, nxd, nrd, nphi)
            ratio = sino_for_reconstruction / (fpsino + 1.0e-9)
            correction = back_proj_system_torch(ratio, syst_mat, nxd, nrd, nphi) / (self.sens_image+1.0e-9) # sens image with safety offset
            recon = recon * correction
            cv2disp("MLEM", torch_to_np(recon), 0, disp_scale*nxd+15, disp_scale)
            cv2disp("Forward Projection", torch_to_np(fpsino), disp_scale*nxd, disp_scale*nxd+15, disp_scale)
            cv2disp("Ratio", torch_to_np(ratio), disp_scale*(nxd+nrd), 0, disp_scale)
            cv2disp("Correction", torch_to_np(correction), disp_scale*(nxd+nrd), disp_scale*nxd+15, disp_scale)
            print("MLEM", it)
            cv2.waitKey(1)
        return recon


core_iterations = 2

# Instantiate the network class -> an object and load onto the GPU
#deepnet = MLEM_Net(true_sinogram_torch, core_iterations).to(device)
#mlem_recon = deepnet(true_sinogram_torch)
#cv2.waitKey(0)


# -- set up a CNN here ----
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 8, 7, padding=(3,3)), nn.PReLU(),      # 1 image in and 8 images out
            nn.Conv2d(8, 8, 7, padding=(3, 3)), nn.PReLU(),
            nn.Conv2d(8, 8, 7, padding=(3, 3)), nn.PReLU(),
            nn.Conv2d(8, 8, 7, padding=(3, 3)), nn.PReLU(),
            nn.Conv2d(8, 1, 7, padding=(3, 3)), nn.PReLU(),     # 8 images in and 1 image out
        )

    def forward(self, x):
        x = torch.squeeze(self.CNN(x.unsqueeze(0).unsqueeze(0)))    # make sure it's a 4D torch tensor for CNN
        return x

cnn = CNN().to(device)





# with trainable conv net inside for MLEM
class MLEM_CNN_net(nn.Module):
    def __init__(self, cnn, sino_for_reconstruction, num_iter):
        super(MLEM_CNN_net, self).__init__()
        self.num_iter = num_iter
        self.sino_ones = torch.ones_like(sino_for_reconstruction)
        self.sens_image = back_proj_system_torch(self.sino_ones, syst_mat, nxd, nrd, nphi)
        self.cnn = cnn # here

    def forward(self, sino_for_reconstruction):
        recon = torch.ones(nxd, nxd).to(device)
        for it in range(self.num_iter):
            fpsino = forward_proj_system_torch(recon, syst_mat, nxd, nrd, nphi)
            ratio = sino_for_reconstruction / (fpsino + 1.0e-9)
            correction = back_proj_system_torch(ratio, syst_mat, nxd, nrd, nphi) / (
                        self.sens_image + 1.0e-9)  # sens image with safety offset
            recon = recon * correction
            # Inter update cnn
            recon = torch.abs(recon + self.cnn(recon))  # MLEM is strictly positive so abs() is necessary
        cv2disp("MLEM", torch_to_np(recon), 0, disp_scale * nxd + 15, disp_scale)
        cv2disp("Forward Projection", torch_to_np(fpsino), disp_scale * nxd, disp_scale * nxd + 15, disp_scale)
        cv2disp("Ratio", torch_to_np(ratio), disp_scale * (nxd + nrd), 0, disp_scale)
        cv2disp("Correction", torch_to_np(correction), disp_scale * (nxd + nrd), disp_scale * nxd + 15, disp_scale)
        print("MLEM", it)
        cv2.waitKey(1)
        return recon

cnn_mlem = MLEM_CNN_net(cnn, true_sinogram_torch, core_iterations).to(device)
mlemcnn_recon = cnn_mlem(true_sinogram_torch)


# --- training the network here ------
loss_fun = nn.MSELoss()                                         # loss function with MSE
optimiser = torch.optim.Adam(cnn_mlem.parameters(), lr=0.001)   # updating the parameters with adam

train_loss = list() # keep track of the training value
epochs = 2500
for ep in range(epochs):
    rec_out = cnn_mlem(true_sinogram_torch)
    loss = loss_fun(rec_out, torch.squeeze(true_object_torch))
    train_loss.append(loss.item())
    loss.backward()         # find the gradients
    optimiser.step()        # does the update
    optimiser.zero_grad()   # set the gradient to zero

    print('Epoch %d Training loss = %f' % (ep, train_loss[-1]))

cv2.waitKey(0)
