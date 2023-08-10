import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils import measure_covariance_alignment_quality_, align_

class MLP(nn.Module):

    def __init__(self, input_size, hidden_layer_sizes, batch_norm = True, no_hidden_layer = False, N_1 = 0, N_2 = 0, special_dim=2, device=None):
        super().__init__()

        self.nb_layers = len(hidden_layer_sizes)

        if not no_hidden_layer:
            layers = [nn.Linear(input_size, hidden_layer_sizes[0], bias=False), nn.ReLU()]
        else:
            layers = [nn.Linear(input_size, hidden_layer_sizes[0], bias=False)]

        for i in range(self.nb_layers-1):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layer_sizes[i], affine=False))
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1], bias=False))
            if i != self.nb_layers-2:
                layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)

        # Pour les rainbow où l'on souhaite juste une dernière couche entraînable dans l'ex. d'Eldan et Shamir
        self.no_hidden_layer = no_hidden_layer
        self.N_1 = N_1
        self.N_2 = N_2
        self.W_1 = torch.nn.Parameter(torch.Tensor(np.random.normal(size=(special_dim, N_1))), requires_grad=False)
        self.W_2 = torch.nn.Parameter(torch.Tensor(np.random.normal(size=(2, N_2))), requires_grad=False)
        self.C_2 = torch.nn.Parameter(torch.ones(self.N_1).reshape(self.N_1,1)*np.sqrt(2*np.pi)/self.N_1, requires_grad=False)
        self.device = device

    def __call__(self, z):
        if self.no_hidden_layer:
            z = nn.functional.relu(z@self.W_1)@self.C_2
            z = nn.functional.relu(torch.hstack((z.reshape(len(z),1),torch.ones((len(z),1)).to(self.device)))@self.W_2)
        
        if self.training:
            z = self.layers(z)
        else:
            z = nn.functional.softmax(self.layers(z), dim=1)

        return z

    def from_rainbow(self, rainbow_model):
        
        rainbow_layers = [module for module in rainbow_model.modules() if not isinstance(module, nn.Sequential)][1:]
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                layer.weight = nn.Parameter(rainbow_layers[i].weight())
    
    def get_covariances(self):
        covariances = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                covariances.append((layer.weight.T@layer.weight).cpu().detach().numpy())
        return covariances
    
    def get_aligned_covariances(self, X, target_model, device):
        covariances = self.get_covariances()[1:]
        alignment_matrices = self.get_alignment(X, target_model, device)[0]

        aligned_covariances = []
        for i in range(len(covariances)):
            aligned_covariances.append(alignment_matrices[i]@covariances[i]@alignment_matrices[i].T)
        return aligned_covariances
    
    # Distribution des vps de l'autocorrélation des poids 
    def covariances_eigval(self, plot=None):
        covariances = self.get_covariances()
        covariances_eig_val = []
        for covariance in covariances:
            v, V = np.linalg.eigh(covariance)
            covariances_eig_val.append(v)
            if plot=="value":
                plt.hist(v, bins='sqrt')
                plt.show()
            if plot=="rank":
                plt.plot(np.flip(v))
                plt.yscale('log')
                plt.show()
        return covariances_eig_val
    
    def get_alignment(self, X, target_model, device, plot_alignement=-1, prop_train=1):
        # On aligne toutes les activations, pré-SoftMax comprises
        
        alignment_matrices = []
        alignment_quality = []
        
        target_layers = [module for module in target_model.modules() if not isinstance(module, nn.Sequential)][1:]
        
        emb_target = X
        emb_self = emb_target
        cpt_linear = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if cpt_linear>0:
                    align_result = align_(emb_target, emb_self, cpt_linear, plot_alignement=plot_alignement, prop_train=prop_train)
                    alignment_matrices.append(align_result[0])
                    alignment_quality.append(align_result[1])
                cpt_linear += 1
                emb_target = target_layers[i](torch.Tensor(emb_target).to(device)).cpu().detach().numpy()
                emb_self = layer(torch.Tensor(emb_self).to(device)).cpu().detach().numpy()
                if cpt_linear == self.nb_layers:
                    align_result = align_(emb_target, emb_self, cpt_linear, plot_alignement=plot_alignement, prop_train=prop_train)
                    alignment_matrices.append(align_result[0])
                    alignment_quality.append(align_result[1])

            else:
                emb_target = target_layers[i](torch.Tensor(emb_target).to(device)).cpu().detach().numpy()
                emb_self = layer(torch.Tensor(emb_self).to(device)).cpu().detach().numpy()
        
        return alignment_matrices, alignment_quality

    def get_covariance_alignment_quality(self, X, target_model, device):
        covariances = self.get_covariances()[1:]
        target_covariances = target_model.get_covariances()[1:]
        alignment_matrices = self.get_alignment(X, target_model, device)[0]

        l_alignment = []
        for i in range(len(target_covariances)):
            estimate_covariance = alignment_matrices[i]@covariances[i]@alignment_matrices[i].T
            l_alignment.append(measure_covariance_alignment_quality_(estimate_covariance, target_covariances[i]))
        return l_alignment

    def get_bases(self, X, plot = False): # return the PCA bases of each activation layer
        bases = []
        current_emb = X
        cpt = 0
        for layer in self.layers:
            if hasattr(layer, 'C_sqrt'): # si la couche courante est linéaire, alors on récupère les activations
                                         # du niveau précédent
                vps, Vps = np.linalg.eigh(current_emb.T@current_emb)
                bases.append(Vps[:,::-1])
                if plot:
                    fig, axs = plt.subplots(1, 2, tight_layout=True)
                    axs[0].matshow(current_emb.T@current_emb)
                    axs[1].scatter(np.arange(len(vps)), vps[::-1])
                    axs[0].set_title('Activation ' + str(cpt))
                    axs[1].set_title('Activation ' + str(cpt) + ', vps')
                    plt.show()
                cpt += 1
            current_emb = layer(torch.Tensor(current_emb)).detach().numpy()
        return bases