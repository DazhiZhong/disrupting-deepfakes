import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

import defenses.smoothing as smoothing

# from torchvision import transforms
# import PIL

def input_diversity(input_tensor):
    image_resize = 500
    image_width = 128
    image_height = 128
    prob = 0.5

    rnd = int((image_resize - image_width) * torch.rand(()) + image_width)
    rescaled = nn.functional.interpolate(input_tensor,size=rnd,mode='nearest')
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = int(h_rem * torch.rand(()))
    pad_bottom = h_rem - pad_top
    pad_left = int(w_rem * torch.rand(()))
    pad_right = w_rem - pad_left
    padded = nn.functional.pad(rescaled, (pad_left, pad_right,pad_top, pad_bottom),mode='constant',value=0.)
    padded = padded.view((input_tensor.shape[0], image_resize, image_resize,3)).permute(0, 3, 1, 2)
    ret = padded if torch.rand((1))[0] > prob else input_tensor
    ret = nn.functional.interpolate(ret,size=image_height,mode='nearest')

    return ret

def momentum(m, grad, accum):
    grad = grad / torch.norm(grad,1,True)
    accum = m * accum + grad
    return accum

def Adam(grad, accum_g, accum_s, i, beta_1=0.9, beta_2=0.999, alpha=0.01):

    # L_inf norm
    grad_normed = grad / torch.norm(grad,1,True)


    # Adam algorithm
    accum_g = grad_normed * (1-beta_1) + accum_g * beta_1
    accum_s = grad * grad * (1-beta_2) + accum_s * beta_2

    accum_g_hat = accum_g / (1 - (beta_1 ** (i+1)))
    accum_s_hat = accum_s / (1 - (beta_2 ** (i+1)))

    # x = x + optimized_grad
    return accum_g, accum_s, alpha/(torch.pow(accum_s_hat, 0.5) + 1e-6), accum_g_hat



class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=100, a=0.01, feat = None):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.SmoothL1Loss().to(device)
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = True

    def perturb_vanilla(self, X_nat, y, c_trg):
        """
        Vanilla Attack.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()    

        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model(X, c_trg)

            if self.feat:
                output = feats[self.feat]

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat


    def perturb_momentum(self, X_nat, y, c_trg):
        """
        Momentum Attack.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()    


        past_grads = torch.zeros_like(X)
        m = 1.0
        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model(X, c_trg)

            if self.feat:
                output = feats[self.feat]

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            past_grads = momentum(m, grad, past_grads)

            X_adv = X + self.a * past_grads.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat


    def perturb_Adam(self, X_nat, y, c_trg):
        """
        Adam Attack.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()    


        accum_g = torch.zeros_like(X)
        accum_s = torch.zeros_like(X)
        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model(X, c_trg)

            if self.feat:
                output = feats[self.feat]

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            accum_g, accum_s, new_a, grad = Adam(grad, accum_g, accum_s, i)

            X_adv = X + new_a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat



    def perturb_momentum_scaled(self, X_nat, y, c_trg):
        """
        Momentum Attack with scale invariance
        """
        # y = y.to(device=self.device, dtype=torch.int64)
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()    


        past_grads = torch.zeros_like(X)
        m = 1.0
        for i in range(self.k):
            X.requires_grad = True

            grads = torch.zeros_like(X)
            for j in range(5):
                output, feats = self.model(input_diversity(X / (2 ** j)),c_trg)
                if self.feat:
                    output = feats[self.feat]
                self.model.zero_grad()
                loss = self.loss_fn(output,y)
                loss.backward()
                grads += X.grad
            # grads /= 5

            past_grads = momentum(m, grads, past_grads)

            X_adv = X + self.a * past_grads.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat

    def perturb_Adam_scaled(self, X_nat, y, c_trg):
        """
        Adam Attack with scale invariance
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()    


        past_grads = torch.zeros_like(X)
        m = 1.0
        accum_g = torch.zeros_like(X)
        accum_s = torch.zeros_like(X)
        for i in range(self.k):


            grads = torch.zeros_like(X)
            for j in range(-2,2):
                X_temp = X.clone().detach_()
                X_temp.requires_grad = True

                output, feats = self.model(input_diversity(X_temp*(2**j)),c_trg)
                if self.feat:
                    output = feats[self.feat]
                self.model.zero_grad()
                loss = self.loss_fn(output,y)
                loss.backward()
                grads += X_temp.grad
            # past_grads = momentum(m, grads, past_grads)
            accum_g, accum_s, new_a, grad = Adam(grads, accum_g, accum_s, i)
            
            X_adv = X + new_a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat


    def perturb_blur(self, X_nat, y, c_trg):
        """
        White-box attack against blur pre-processing.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  
              
        X_orig = X_nat.clone().detach_()

        # Kernel size
        ks = 11
        # Sigma for Gaussian noise
        sig = 1.5

        # preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks).to(self.device)
        preproc = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks).to(self.device)

        # blurred_image = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks).to(self.device)(X_orig)
        blurred_image = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks).to(self.device)(X_orig)

        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model.forward_blur(X, c_trg, preproc)

            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat, blurred_image

    def perturb_blur_iter_full(self, X_nat, y, c_trg):
        """
        Spread-spectrum attack against blur defenses (gray-box scenario).
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  

        # Gaussian blur kernel size
        ks_gauss = 11
        # Average smoothing kernel size
        ks_avg = 3
        # Sigma for Gaussian blur
        sig = 1
        # Type of blur
        blur_type = 1

        for i in range(self.k):
            # Declare smoothing layer
            if blur_type == 1:
                preproc = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks_gauss).to(self.device)
            elif blur_type == 2:
                preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks_avg).to(self.device)

            X.requires_grad = True
            output, feats = self.model.forward_blur(X, c_trg, preproc)

            if self.feat:
                output = feats[self.feat]

            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            # Iterate through blur types
            if blur_type == 1:
                sig += 0.5
                if sig >= 3.2:
                    blur_type = 2
                    sig = 1
            if blur_type == 2:
                ks_avg += 2
                if ks_avg >= 11:
                    blur_type = 1
                    ks_avg = 3

        self.model.zero_grad()

        return X, X - X_nat

    def perturb_blur_eot(self, X_nat, y, c_trg):
        """
        EoT adaptation to the blur transformation.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  

        # Gaussian blur kernel size
        ks_gauss = 11
        # Average smoothing kernel size
        ks_avg = 3
        # Sigma for Gaussian blur
        sig = 1
        # Type of blur
        blur_type = 1

        for i in range(self.k):
            full_loss = 0.0
            X.requires_grad = True
            self.model.zero_grad()

            for j in range(9):  # 9 types of blur
                # Declare smoothing layer
                if blur_type == 1:
                    preproc = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks_gauss).to(self.device)
                elif blur_type == 2:
                    preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks_avg).to(self.device)

            
                output, feats = self.model.forward_blur(X, c_trg, preproc)
            
                loss = self.loss_fn(output, y)
                full_loss += loss

                if blur_type == 1:
                    sig += 0.5
                    if sig >= 3.2:
                        blur_type = 2
                        sig = 1
                if blur_type == 2:
                    ks_avg += 2
                    if ks_avg >= 11:
                        blur_type = 1
                        ks_avg = 3
                
            full_loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat




    def perturb_iter_class(self, X_nat, y, c_trg):
        """
        Iterative Class Conditional Attack
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  

        j = 0
        J = len(c_trg)

        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model(X, c_trg[j])

            self.model.zero_grad()

            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            j += 1
            if j == J:
                j = 0

        return X, eta


    def perturb_joint_class(self, X_nat, y, c_trg):
        """
        Joint Class Conditional Attack
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  

        J = len(c_trg)
        
        for i in range(self.k):
            full_loss = 0.0
            X.requires_grad = True
            self.model.zero_grad()

            for j in range(J):
                output, feats = self.model(X, c_trg[j])

                loss = self.loss_fn(output, y)
                full_loss += loss

            full_loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        return X, eta

def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res

def perturb_batch(X, y, c_trg, model, adversary):
    # Perturb batch function for adversarial training
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    
    adversary.model = model_cp

    X_adv, _ = adversary.perturb(X, y, c_trg)

    return X_adv
