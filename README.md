# Stage_4A_MVA

## Présentation

Ce répertoire présente les codes associés au rapport de stage "Modélisation probabiliste des réseaux de neurones et séparation par la profondeur".
Ils se répartissent comme suit :
- MLP.py, classe MLP ;
- rainbow_MLP.py, classe rainbow_MLP ;
- testing.py, test des modèles ;
- training.py, entraînement des MLPs ;
- utils.py, calculs divers et affichage.

Les notebooks fournis permettent de reproduire les expériences du rapport :
- Convergence_activations_covariances.ipynb, pour la partie 2.4.2 (figure 1) ;
- Spectres_covariances.ipynb, pour les parties 2.4.2 et 2.4.3 (figures 2 et 5) ;
- Echantillonnage_gaussien.ipynb, pour la partie 2.4.3 (figures 3, 4 et 6) ;
- Generation_Eldan_Shamir.ipynb, pour la génération des données de la partie 3.2.2 (tirés de Sun et al. (2018), https://github.com/syitong/randrelu) ;
- Sep_profondeur.ipynb, pour la partie 2.4.2 (figure 9) ;
- Visualisation_frontieres.ipynb, pour l'annexe A.2  (figures 10-13).

## Utilisation

Les notebooks peuvent être utilisés directement, et sont indépendants - à l'exception de Sep_profondeur.ipynb qui requiert d'avoir exécuté Generation_Eldan_Shamir.ipynb au préalable si l'on utilisé une dimension d'entrée différente de 5 (pour d=5, les données sont directements fournis au téléchargement dans 'data/').

## Licences

Le répertoire proposé ici est placé sous licence Apache 2.0. Le répertoire de Sun et al. (2018) est, quant à lui, sous licence MIT.
