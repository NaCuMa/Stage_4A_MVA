import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from scipy.linalg import sqrtm
from utils import align_

# On ré-implémente la classe "Linear" de façon à pouvoir réaliser la décomposition des poids en bruit blanc et covariance 
class Linear(nn.Module):
    def __init__(self, n_in, n_out, bias = True):
        super().__init__()

        # Initialisation aléatoire de la covariance "self.C_sqrt"
        init_C = np.random.randn(n_in, n_in)
        self.C_sqrt = nn.Parameter(torch.Tensor(sqrtm(init_C@init_C.T)), requires_grad = False)
        #self.C_sqrt = nn.Parameter(torch.eye(n_in), requires_grad = False)

        ## Initialisation des alignements "self.A"
        self.A = nn.Parameter(torch.eye(n_in), requires_grad = False)

        # Initialisation aléatoire du bruit blanc "self.G"
        self.G = nn.Parameter(torch.randn(n_out, n_in), requires_grad = False)
        
        self.n_in = n_in
        self.n_out = n_out

        # On offre la possibilité d'ajouter un biais, bien que le modèle n'en contienne pas par défaut
        self.bias_bool = bias
        if self.bias_bool:
            self.bias = nn.Parameter(torch.zeros(n_out), requires_grad = False)
            
    def weight(self):
        return self.G@self.C_sqrt@self.A
    
    def resample(self):
        # Le ré-échantillonage de la couche revient à tirer un nouveau bruit blanc "self.G"
        self.G = nn.Parameter(torch.randn(self.G.shape[0], self.G.shape[1]), requires_grad = False)
    
    def forward(self, x):
        # (bs, n_in) @ ((n_out, n_in)@(n_in, n_in)@(n_in, n_in)).T = (bs, n_out), la transposition étant
        # due au fait que les réalisations sont stockés en lignes et non en colonnes
        Wx = x @ self.weight().T

        # Ajout du biais le cas échéant
        if self.bias_bool:
            return Wx + self.bias
        return Wx

# A partir du bloc précédent, on construit le modèle des rainbow MLPs
class rainbow_MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, batch_norm = True):
        # "hidden_layer_sizes" est une liste donnant la largeur des différentes couches composant le MLP,
        # dernière incluse (i.e. si on considère une tâche de classification à k-classes, hidden_layer_sizes[-1] = k)
        super().__init__()

        self.nb_layers = len(hidden_layer_sizes)

        layers = [Linear(input_size, hidden_layer_sizes[0], bias=False), nn.ReLU()]
    
        for i in range(self.nb_layers-1):
            # Si on ajoute une batch norm, on le fait dtsq les activations à aligner soient normalisées
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layer_sizes[i], affine=False, momentum=None))

            layers.append(Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1], bias=False))
            if i!=self.nb_layers-2:
                layers.append(nn.ReLU())

        # Comme on s'intéresse à une tâche de classification multiclasse, on termine par un Softmax
        layers.append(nn.Softmax(dim=1))
        
        self.layers = nn.Sequential(*layers)

    def __call__(self, z):
        return self.layers(z)
    
    ## Les deux fonctions suivantes permettent le ré-échantillonnage d'un MLP entraîné (classe MLP) :
    # "approx" estime les covariances des poids de ce réseau et met à jour les attributs "C_sqrt" des couches en conséquence,
    # "resample" ré-échantillonne les bruits blancs "G" et réaligne les activations couche par couche

    def approx(self, normal_MLP):        
        rainbow_layers = [module for module in self.modules() if not isinstance(module, nn.Sequential)][1:]
        
        cpt_linear = 0
        for i, normal_layer in enumerate(normal_MLP.layers):
            if isinstance(normal_layer, nn.Linear):
                W = normal_layer.weight.detach().numpy()

                rainbow_layers[i].C_sqrt = nn.Parameter(torch.Tensor(np.real(sqrtm(W.T@W/W.shape[0]))), requires_grad = False)
                rainbow_layers[i].G = nn.Parameter(torch.randn(rainbow_layers[i].G.shape[0],
                                                rainbow_layers[i].C_sqrt.shape[0]), requires_grad = False)
                
                cpt_linear += 1
                if cpt_linear == self.nb_layers:
                    rainbow_layers[i].G = nn.Parameter(torch.Tensor(W), requires_grad = False)

    # Ré-échantillonnage
    def resample(self, X, normal_MLP, plot_alignement = -1):
        
        alignment_quality = []
        cpt_linear = 0
        
        normal_MLP_layers = [module for module in normal_MLP.modules() if not isinstance(module, nn.Sequential)][1:]
        normal_MLP_layers.append(nn.Softmax(dim=1))

        emb_target = X
        emb_resampled = emb_target

        self.train() # On souhaite estimer les paramètres des BNs

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'C_sqrt'):
                if cpt_linear != 0:
                    align_result = align_(emb_target, emb_resampled, cpt_linear, plot_alignement)
                    A_hat = align_result[0]*np.sqrt(emb_target.shape[1]/emb_resampled.shape[1])
                    alignment_quality.append(align_result[1])

                    layer.A = nn.Parameter(torch.Tensor(A_hat), requires_grad = False)
                    if cpt_linear != self.nb_layers-1:
                        layer.resample()
                    else:
                        layer.C_sqrt = nn.Parameter(torch.eye(layer.C_sqrt.shape[0]), requires_grad = False)
                        layer.A = nn.Parameter(torch.Tensor(A_hat), requires_grad = False)
                else:
                    layer.resample()
                cpt_linear += 1
                emb_target = normal_MLP_layers[i](torch.Tensor(emb_target)).detach().numpy()
                emb_resampled = layer(torch.Tensor(emb_resampled)).detach().numpy()
            else:
                if isinstance(layer, nn.BatchNorm1d):
                    layer.reset_running_stats() # si on ne réinitialise pas, la moyenne sera faite avec les stats de l'échantillonnage précédent
                emb_target = normal_MLP_layers[i](torch.Tensor(emb_target)).detach().numpy()
                emb_resampled = layer(torch.Tensor(emb_resampled)).detach().numpy()
        
        self.eval() # On fixe les BNs

        return alignment_quality
    
    
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
    
    def C_identity(self):
        for i, layer in enumerate(self.layers):
                if hasattr(layer, 'C_sqrt'):
                    new_C_sqrt = torch.eye(layer.C_sqrt.shape[0])
                    layer.C_sqrt = nn.Parameter(new_C_sqrt/np.linalg.norm(new_C_sqrt.detach().numpy())*np.linalg.norm(layer.C_sqrt.detach().numpy()))
        return
