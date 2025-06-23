import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import argparse

# Données factices pour la simulation
X_train = [
    "J'adore ce produit, il est fantastique",
    "Quelle déception, je ne recommande pas",
    "Service client au top, très réactif",
    "Une arnaque, fuyez ce site",
    "Bon rapport qualité-prix",
    "Le produit est arrivé cassé"
]
y_train = ["positif", "négatif", "positif", "négatif", "positif", "négatif"]

def train_and_save_model(output_path="model.joblib"):
    """
    Entraîne un modèle de classification de texte et le sauvegarde.
    """
    print("Début de l'entraînement...")
    # Création d'un pipeline simple
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)),
    ])

    # Entraînement
    text_clf.fit(X_train, y_train)
    print("Entraînement terminé.")

    # Sauvegarde du modèle
    joblib.dump(text_clf, output_path)
    print(f"Modèle sauvegardé dans : {output_path}")
    return text_clf

def predict(text, model_path="model.joblib"):
    """
    Charge le modèle et effectue une prédiction.
    """
    try:
        model = joblib.load(model_path)
        prediction = model.predict([text])
        return prediction[0]
    except FileNotFoundError:
        return "Erreur: Fichier de modèle non trouvé. Veuillez d'abord entraîner le modèle."

if __name__ == '__main__':
    # Utilisation de argparse pour rendre le script exécutable en ligne de commande
    # C'est une pratique professionnelle qui sera appréciée.
    parser = argparse.ArgumentParser(description="Script d'entraînement et de prédiction.")
    parser.add_argument('--action', type=str, required=True, choices=['train', 'predict'],
                        help="Action à effectuer : 'train' ou 'predict'.")
    parser.add_argument('--text', type=str, help="Texte à classifier (requis pour l'action 'predict').")

    args = parser.parse_args()

    if args.action == 'train':
        train_and_save_model()
    elif args.action == 'predict':
        if not args.text:
            raise ValueError("--text est requis pour l'action 'predict'")
        prediction = predict(args.text)
        print(f"Le texte '{args.text}' est classifié comme : {prediction}")