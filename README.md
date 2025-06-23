# Mini-Projet de Déploiement d'IA

Ce projet implémente un modèle de Machine Learning simple (ex: classification de texte sur des données factices) et met en place un pipeline CI/CD complet avec GitHub Actions pour le déploiement automatique sur Hugging Face Hub.

## Modèle
Le modèle est un `Pipeline` scikit-learn combinant un `TfidfVectorizer` et un classifieur `SGDClassifier`. Il est entraîné sur un jeu de données textuelles simple et sauvegardé au format `.joblib`.

## Déploiement
Le déploiement est automatisé via un workflow GitHub Actions qui se déclenche à chaque push sur la branche `main`.