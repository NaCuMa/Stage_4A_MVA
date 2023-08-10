import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Blocs de calcul 
def apply(model, dataloader, device):
    y_true  = []
    y_pred = []

    for X, y in dataloader:
        X = X.to(device)
        y_true.extend(y.detach().numpy().tolist())
        y_pred.extend(model(X).cpu().detach().numpy().tolist())
    return y_true, y_pred

def align_(emb_target, emb, couche, plot_alignement=-1, prop_train=1):

    if prop_train == 1:
        emb_target_train = emb_target
        emb_target_test = emb_target

        emb_train = emb
        emb_test = emb
    else:
        emb_target_train = emb_target[:int(prop_train*emb_target.shape[0])]
        emb_target_test = emb_target[int(prop_train*emb_target.shape[0]):]

        emb_train = emb[:int(prop_train*emb.shape[0])]
        emb_test = emb[int(prop_train*emb.shape[0]):]

    U, S, VT = np.linalg.svd(emb_target_train.T@emb_train, full_matrices=False)
    A_hat_train = U@VT

    A_estimate = emb_test@A_hat_train.T*np.sqrt(emb_target.shape[1]/emb.shape[1])
    if plot_alignement!=-1: # Affichage
        plot_alignement_(A_estimate, emb_target_test, couche, plot_alignement)

    return A_hat_train, measure_activation_alignment_quality_(A_estimate, emb_target_test)

# Reçoit deux matrices (nb_echantillons, dim_activations) et calcule la moyenne sur les
# échantillons de la norme de la différence entre les activations (normalisée par la dimension des activations)
def measure_activation_alignment_quality_(A_hat_phi, phi, plot=False):
    delta_phi = np.linalg.norm((A_hat_phi-phi)/np.sqrt(phi.shape[1]), axis = 1)

    alignement_quality_mean = np.mean(delta_phi)
    alignement_quality_var = np.var(delta_phi)

    if plot:
        plt.hist(delta_phi, bins='sqrt')
        plt.show()
    return alignement_quality_mean, alignement_quality_var

# Mesure la qualité de l'alignement entre les covariances de deux MLPs entraînés
def measure_covariance_alignment_quality_(estimate_covariance, target_covariances, order=np.inf):
    return np.linalg.norm(target_covariances-estimate_covariance, ord=order)/np.linalg.norm(target_covariances, ord=order)

# Renvoi la densité de la distribution de MP
def Marchenko_Pastur(x, lambda_, sigma2=1):
    result = np.zeros(len(x))
    lambda_plus=sigma2*(1+lambda_**0.5)**2
    lambda_minus=sigma2*(1-lambda_**0.5)**2
    well_def_indices = np.where((x<=lambda_plus) & (x>=lambda_minus))[0]
    well_def_x = x[well_def_indices]
    result[well_def_indices] = np.sqrt((lambda_plus-well_def_x)*(well_def_x-lambda_minus))/(2*np.pi*sigma2*lambda_*well_def_x)
    return result

## Visualisation -------------------------------------------------------------------------

def plot_alignement_(A_estimate, emb, couche, coord):
    plt.plot(A_estimate[coord], alpha=0.5, label = 'Activations réalignées')
    plt.plot(emb[coord], alpha=0.5, label = "Activations d'origine")
    plt.title("Alignement de \phi_"+str(couche)+"(x["+str(coord)+"])")
    plt.legend(loc='best')
    plt.xlabel("Coordonnées du plongement")
    plt.show()
    print("L'écart d'alignement entre les activations affichées est de : ", measure_activation_alignment_quality_(A_estimate[coord:coord+1], emb[coord:coord+1])[0])

    return

# Affichage des frontières de décision des classifieurs entrainé et ré-échantillonné
def plot_results_2D(X, y, model_trained, model_sampled):
    # On vérifie que les données sont bien en 2D
    assert X.shape[1]==2, "Les données d'entrées sont dans R_"+str(X.shape[1])+", pas dans R_2."

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[25, 10])

    colors = ['dodgerblue', 'crimson']
    ax1.plot(X[:, 0][y == 0], X[:, 1][y == 0], "o", markersize = 2, color=colors[0])
    ax1.plot(X[:, 0][y == 1], X[:, 1][y == 1], "^", markersize = 2, color=colors[1])
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    pred_trained = model_trained(torch.Tensor(np.c_[(xx.ravel(), yy.ravel())]))[:,1]
    ax1.contour(xx, yy, pred_trained.detach().numpy().reshape(xx.shape), levels=14, linewidths=0.5, colors='k')
    cntr1 = ax1.contourf(xx, yy, pred_trained.detach().numpy().reshape(xx.shape), levels=14, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax1)
    ax1.set_title("Modèle entrainé", fontsize=20)
    
    ax2.plot(X[:, 0][y == 0], X[:, 1][y == 0], "o", markersize = 2, color=colors[0])
    ax2.plot(X[:, 0][y == 1], X[:, 1][y == 1], "^", markersize = 2, color=colors[1])
    pred_sampled = model_sampled(torch.Tensor(np.c_[(xx.ravel(), yy.ravel())]))[:,1]
    ax2.contour(xx, yy, pred_sampled.detach().numpy().reshape(xx.shape), levels=14, linewidths=0.5, colors='k')
    cntr2 = ax2.contourf(xx, yy, pred_sampled.detach().numpy().reshape(xx.shape), levels=14, cmap="RdBu_r")
    fig.colorbar(cntr2, ax=ax2)
    ax2.set_title("Modèle ré-échantillonné", fontsize=20)
    
    plt.tight_layout()
    plt.show()
    return

def plot_spectra(cov_matrix, plot = "value"):
    v, V = np.linalg.eigh(cov_matrix)
    if plot=="value":
        plt.hist(v, bins='sqrt')
        plt.show()
    if plot=="rank":
        plt.plot(np.flip(v))
        plt.yscale('log')
        plt.show()
    return

## Vérification -------------------------------------------------------------------------

def check_alignement(X, model_trained, model_sampled, threshold_alignment=0.1, plot_sample=-1, plot_distrib=False):
    sampled_layers = [module for module in model_sampled.modules() if not isinstance(module, nn.Sequential)][1:]

    z_trained = X
    z_sampled = X

    cpt_linear = 0
    alignment_quality = []
    for i, layer in enumerate(model_trained.layers):
        if hasattr(sampled_layers[i], 'C_sqrt') :
            z_sampled = z_sampled@sampled_layers[i].A.T
            if cpt_linear>0:
                alignment_quality.append(measure_activation_alignment_quality_(z_sampled.detach().numpy(),
                                        z_trained.detach().numpy(), plot=plot_distrib))
                assert alignment_quality[-1][0] < threshold_alignment, "Alignements incohérents ("+str(alignment_quality[-1][0])+"<"+str(threshold_alignment)+"*1)"
                if plot_sample!=-1:
                    plot_alignement_(z_sampled.detach().numpy(), z_trained.detach().numpy(), cpt_linear, plot_sample)

            z_sampled = z_sampled@sampled_layers[i].C_sqrt.T
            z_sampled = z_sampled@sampled_layers[i].G.T
            cpt_linear+=1
        else:
            z_sampled = sampled_layers[i](z_sampled)
        z_trained = layer(z_trained)
    return alignment_quality

# On vérifie, pour les réseaux ayant des BNs, si les activations (post-BNs) sont bien normalisées
def check_normalisation(model, X, plot=False, threshold_mean=0.1, threshold_var=0.1):
    z = X
    for layer in model.layers:
        if hasattr(layer, 'running_mean'):

            z_mean = np.mean(z.detach().numpy(), axis=0)
            delta_mean = z_mean-layer.running_mean.detach().numpy()

            z_var = np.var(z.detach().numpy(), axis=0)
            delta_var = z_var-layer.running_var.detach().numpy()
            
            if plot:
                plt.plot(delta_mean)
                lim_mean_vector = np.full(z.shape[1], threshold_mean*np.linalg.norm(z_mean)/np.sqrt(z.shape[1]))
                plt.plot(lim_mean_vector, c='r')
                plt.plot(-lim_mean_vector, c='r')
                plt.show()
                plt.plot(delta_var)
                lim_var_vector = np.full(z.shape[1], threshold_var*np.linalg.norm(z_var)/np.sqrt(z.shape[1]))
                plt.plot(lim_var_vector, c='r')
                plt.plot(-lim_var_vector, c='r')
                plt.show()

            assert np.linalg.norm(delta_mean) < threshold_mean*np.linalg.norm(z_mean), "Moyennes incohérentes ("+str(np.linalg.norm(delta_mean))+"<"+str(threshold_mean*np.linalg.norm(z_mean))
            assert np.linalg.norm(delta_var) < threshold_var*np.linalg.norm(z_var), "Variances incohérentes ("+str(np.linalg.norm(delta_var))+"<"+str(threshold_var*np.linalg.norm(z_var))

        z = layer(z)
    return