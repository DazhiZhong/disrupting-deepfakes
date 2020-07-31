import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

def input_diversity(input_tensor):
    image_resize = 500
    image_width = 128
    image_height = 128
    prob = 0.5

    rnd = int((image_resize - image_width)*torch(())+image_width)
    rescaled = nn.functional.interpolate(input_tensor,size=rnd,mode='nearest')
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = int(h_rem*torch.rand(()))
    pad_bottom = h_rem - pad_top
    pad_left = int(w_rem*torch.rand(()))
    pad_right = w_rem-pad_left
    padded = nn.functional.pad(rescaled,(pad_left,pad_right,pad_top,pad_bottom),mode='constant',value=0.)
    padded = padded.view((input_tensor.shape[0],image_resize,image_resize,3)).permute(0,3,1,2)
    ret = padded if torch.rand((1))[0] > prob else input_tensor
    ret = nn.functional.interpolate(ret,size=image_height,mode='nearest')

def momentum(m,g,accum_g):
    g = g / torch.norm(g,1,True)
    accum_g = m * accum_g + g
    return accum

def Adam(g,accum_g,accum_s,i,beta_1=0.9,beta_2=0.999,alpha=0.01):
    g_normed = g / torch.norm(g,1,True)
    accum_g = g_normed * (1-beta_1) + accum_g * beta_1
    accum_s = g * g * (1 - beta_2) + accum_s * beta_2
    accum_g_hat = accum_g / (1 - (beta_1 ** (i+1)))
    accum_s_hat = accum_s / (1 - (beta_2 ** (i+1)))
    return accum_g, accum_s, alpha/(torch.pow(accum_s_hat, 0.5) + 1e-6), accum_g_hat


class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01):
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

        # PGD or I-FGSM?
        self.rand = True

    def perturb(self, X_nat, y, c_trg):
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
            output_att, output_img = self.model(X, c_trg)

            out = imFromAttReg(output_att, output_img, X)

            self.model.zero_grad()

            # Attention attack
            # loss = self.loss_fn(output_att, y)

            # Output attack
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(out, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            # Debug
            # X_adv, loss, grad, output_att, output_img = None, None, None, None, None

        return X, eta

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

        for i in range(self.k):
            X.requires_grad = True
            output_att, output_img = self.model(X, c_trg)

            out = imFromAttReg(output_att, output_img, X)

            self.model.zero_grad()

            # Attention attack
            # loss = self.loss_fn(output_att, y)

            # Output attack
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(out, y)
            loss.backward()
            grad = X.grad

            past_grads = momentum(m,grad,past_grads)

            X_adv = X + self.a * past_grads.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            # Debug
            # X_adv, loss, grad, output_att, output_img = None, None, None, None, None

        return X, eta
    
    def perturb_adam(self, X_nat, y, c_trg):
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
            output_att, output_img = self.model(X, c_trg)

            out = imFromAttReg(output_att, output_img, X)

            self.model.zero_grad()

            # Attention attack
            # loss = self.loss_fn(output_att, y)

            # Output attack
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(out, y)
            loss.backward()
            grad = X.grad

            accum_g, accum_s, new_a, grad = Adam(grad,accum_g,accum_s,i)

            X_adv = X + new_a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            # Debug
            # X_adv, loss, grad, output_att, output_img = None, None, None, None, None

        return X, eta



    def perturb_iter_class(self, X_nat, y, c_trg):
        """
        Iterative Class Conditional Attack
        """
        X = X_nat.clone().detach_()

        j = 0
        J = c_trg.size(0)

        for i in range(self.k):
            X.requires_grad = True
            output_att, output_img = self.model(X, c_trg[j,:].unsqueeze(0))

            out = imFromAttReg(output_att, output_img, X)

            self.model.zero_grad()

            # loss = self.loss_fn(output_att, y)
            loss = self.loss_fn(out, y)
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
        X = X_nat.clone().detach_()

        J = c_trg.size(0)
        
        for i in range(self.k):
            full_loss = 0.0
            X.requires_grad = True
            self.model.zero_grad()

            for j in range(J):
                output_att, output_img = self.model(X, c_trg[j,:].unsqueeze(0))

                out = imFromAttReg(output_att, output_img, X)

                # loss = self.loss_fn(output_att, y)
                loss = self.loss_fn(out, y)
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

def imFromAttReg(att, reg, x_real):
    """Mixes attention, color and real images"""
    return (1-att)*reg + att*x_real