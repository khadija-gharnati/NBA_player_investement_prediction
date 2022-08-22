Ce projet propose un programme de prédiction concu pour aider les investisseurs NBA à choisir les joueur sur qui il faut investir.
Ce projet se compose des fichiers suivant :
    0 - nba_logreg.csv : le dataset initial.
    1 - le notebook Analyse.ipynb : explique les différantes étapes d'analyses qui nous ont amené au choix du modèle final.
    2 - le fichier test.py : sa fonction est de choisir, entrainer et évaluer le modèle de classification et le stocker dans Prediction_model.
    3 - Prediction_model : contient le modèle de prédiction finale.
    4 - deploy.py : permet de déployer le modèle et génèrer une interface utilisateur facile à utiliser.
    5 - le dossier templates : stocke le code html de la page générée par deploy.py.
    6 - le dossier static : contient l'image du background ainsi que le code CSS de la page générée par deploy.py.
    7 - le fichier reduirement.txt : liste les librairies nécessaires pour le fonctionnement du code.



Pour obtenir l'interface utilisable il faut suivre les étapes suivante (sur une machine windows) :
    1 - dans une invite de commandes : cd "chemin du projet",
                                        pip install -r requirements.txt,
                                        python test.py
                                        python deploy.py ==> url
    2 - copiez l'url affichée dans l'invite de commande et ouvrez la dans un navigateur web
    3 - le programme est pret à etre utilisé