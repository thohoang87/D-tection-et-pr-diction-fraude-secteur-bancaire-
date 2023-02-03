# Détection et prédiction de la fraude dans le secteur bancaire

## 1 Introduction :  

La détection de fraude est une problématique courante dans de nombreux domaines, notamment les banques et le secteur financier, les assurances, dans le domaine social, judiciaire, et bien d'autres encore.  Au cours des dernières années, les tentatives de fraude ont connu une forte recrudescence, ce qui rend la lutte contre ce phénomène plus importante que jamais. 
     
Afin de réaliser ces détections, les banques se focalisent aujourd’hui sur l’application de techniques d’intelligences artificielles. Ces algorithmes, qu’ils soient supervisés ou non, permettent d’identifier des comportements atypiques et suspects. 
     
Grâce aux différentes techniques et outils avancés de Machine Learning et/ou Deep Learning, nous pouvons détecter rapidement mais également anticiper les actes de fraude et prendre des mesures immédiates pour limiter leur impact financier. Du coup, ceci permet aux banques d’améliorer grandement leurs techniques de traitement des données, et indirectement de multiplier les bénéfices.
    
Cependant, la difficulté principale repose sur le déséquilibre des classes. Selon le rapport annuel 2019 de l’OSMP (2020), les transactions bancaires frauduleuses ne représentent que 0.001% des transactions, alors que le montant des transactions frauduleuses représente 1% des montants de transactions. Les données déséquilibrées complexifient les analyses prédictives et posent un véritable problème en Machine Learning et Deep Learning.
     
Afin de résoudre ces problèmes de déséquilibre des classes, nous optons pour  des méthodes de rééchantillonnage (re-sampling) qui consistent à modifier la distribution des données avant d’entraîner le modèle prédictif. Cela permet d’équilibrer les données pour faciliter la prédiction.
      
L’objectif de cette étude est donc de mettre au point un modèle performant de détection de fraude bancaire.  
 
## 2 Analyse des données 
### 2.1 Description des données 
 
Les données sur lesquelles nous travaillons sont des données réelles. Elles sont issues d’une enseigne de la grande distribution ainsi que de certains organismes bancaires (FNCI et Banque de France). Chaque ligne représente une transaction effectuée par chèque dans un magasin de l’enseigne quelque part en France. Notre étude est basée sur jeu de données avec deux composantes principales : 

- La variable cible FlagImpaye : c’est La variable à prédire. Il s’agit d’une variable qui ne peut prendre que deux valeurs possibles : 0 la transaction est acceptée et considérée comme "normale", 1 la transaction est refusée car considérée comme "frauduleuse". 

- Les variables explicatives : On peut retrouver par exemple montant de la transaction, date de la transaction… 

### 2.2 Analyse synthétique :  
 
Tableau Statistiques descriptives pour les variables : voir tableau des statistiques sur Python. 
Répartition de la distribution entre les deux classes de la variable cible « FlagImpaye » :  
 
  <img width="265" alt="image" src="https://user-images.githubusercontent.com/114235978/216571853-6f64e1df-88af-4b06-8e68-19269eac6fc5.png">

On remarque bien qu'il y a un déséquilibre entre les deux classes. La variable cible a plus d'observations dans la classe d’acceptation de transactions. 
 
### Graphique de corrélation : 

 <img width="454" alt="image" src="https://user-images.githubusercontent.com/114235978/216572013-b89f399a-3e21-4e95-af0d-e1ef52ee142d.png">
 
### Corrélation positive : 

D’après la matrice de corrélation, qui mesure le degré de relation linéaire entre chaque paire de variables, on constate que les variables les plus corrélées positivement avec la variable cible "FlagImpaye" sont VérifianceCPT2, VérifianceCPT3 et VérifianceCPT1. Elles désignent respectivement le nombre de transactions effectuées par le même identifiant bancaire au cours des trois derniers jours, le nombre de transactions effectuées par le même identifiant bancaire au cours des sept derniers jours et nombre de transactions effectuées par le même identifiant bancaire au cours du même jour. Ceci est interprétable comme suit : plus le nombre de transactions effectuées par le même identifiant bancaire au cours des trois derniers jours est élevé plus la transaction est susceptible d'être frauduleuse.  

### Corrélation négative :  
Les variables les moins corrélées avec la variable cible "FlagImpaye" sont ScoringFP1, ScoringFP2 et ScoringFP3. 

La relation entre ScoringFP1 et la variable cible est négative. Plus le ScoringFP1 est élevé moins il y a risque de fraude.  

On aurait aimé tester d’autres types de corrélations mais ça ne fonctionne pas en raison de la volumétrie des données. 

## 3 Méthodologie : 
### 3.1 Les algorithmes de rééchantillonnages : 
 
Les données déséquilibrées (ou imbalanced data) sont un problème fréquemment rencontré dans les modèles de classification, qu’il s’agisse de classification binaire ou de classification multi-classes. Dans notre cas il s’agit d’une classification binaire (acceptation ou refus de transaction). 
On peut parler de données déséquilibrées dès lors que les deux classes ne sont pas présentes avec la même fréquence dans les données, i.e. que le ratio n’est pas 50%/50%. Mais en pratique on ne parle de données déséquilibrées qu’à partir du moment où le déséquilibre dépasse 10%/90%.  

Dans ce cas de détection de fraude, on remarque que 99.12% des transactions effectuées sont valides, et seulement 0.88% frauduleuses. Les transactions valides sont alors appelées la classe majoritaire, et les fraudes la classe minoritaire. 

Ceci pose de réelles difficultés aux algorithmes de Machine Learning et de Deep Learning et conduit surtout au sur apprentissage. 
L’une des solutions pour traiter les données déséquilibrées est de les “rééquilibrer”. Ce type d’approches – appelées data-level solutions – se décline sous 2 formes principales : 

- Le sur-échantillonnage (oversampling) : Le nombre d’individus minoritaires est augmenté pour qu’ils aient plus d’importance lors de la modélisation. Différentes solutions sont possibles, comme le SMOTE, Adasyn, Random, etc. Pour notre cas, on a opté pour le SMOTE qui consiste à synthétiser des éléments de la classe minoritaire, sur la base de ceux qui existent déjà. Il sélectionne aléatoirement un point de la classe minoritaire et calcul les k-voisins les plus proches pour ce point. Les points synthétiques sont ajoutés entre le point choisi et ses voisins. 
  
- Le sous-échantillonnage (undersampling) : Parmi les individus majoritaires, on en retire une partie afin d’accorder plus d’importance aux individus minoritaires. Cette approche permet de diminuer la redondance des informations apportées par le grand nombre d’individus majoritaires. Pour notre cas, on a opté pour le Random Under Sampler. 
 
### 3.2 Algorithmes de prédiction utilisée : 
- Random Forest

- Modèle NN : Perceptron multicouche 

- XGBoost

### 3.3 Mesures de performance : 

Une autre solution pour améliorer les performances des algorithmes sur des jeux de données déséquilibrés est de travailler sur la métrique de validation. Pour la détection de fraude, plutôt que d’utiliser l’accuracy, nous utiliserons le F1-score pour évaluer la performance des différents algorithmes et techniques de classification utilisées dans notre démarche.

Dans ce cas, il est plus intéressant que l’accuracy car le nombre de vrais négatifs (TN) n’est pas pris en compte. Et dans les situations d’imbalanced class, nous avons une majorité de vrais négatifs qui faussent complètement notre perception de la performance de l’algorithme. Un grand nombre de vrais négatifs (TN) laissera le F1-Score de marbre. 

Le F1-score permet de résumer les valeurs de la precision et du recall en une seule métrique. Mathématiquement, le F1-score est défini comme étant la moyenne harmonique de la precision et du recall, ce qui se traduit par l’équation suivante : 

𝐹1-𝑠𝑐𝑜𝑟𝑒 = 2 x (Precision × Rappel)/(Precision + Rappel)

Avec la Précision définit comme TN/(TN+FN) et le rappel comme  TN/(TN+FP)

Et la matrice de confusion est définie somme suit :

<img width="514" alt="Capture d’écran 2023-02-03 à 11 10 50" src="https://user-images.githubusercontent.com/114235978/216573335-d20be5a3-523e-4250-a907-96c842b476f8.png">

Dans notre cas on cherche à minimiser le taux d’erreur sur la classe positive (FP) et à maximiser le F1-score, l’enjeu est de trouver un compromis entre les deux. 

𝑇𝑎𝑢𝑥 d’erreur 𝑠𝑢𝑟 𝑙𝑎 𝑐𝑙𝑎𝑠𝑠𝑒 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒 = FP/(FP+TP)

Ce qui nous intéresse dans ce cas c’est le fait de prédire les fraudes (les cas positifs) c’est pour cette raison qu’on s’intéresse à la mesure du taux d’erreur de la classe positive.  

## 4 Expériences : 

Avant d’analyser les données et d’entraîner les modèles, il convient de retraiter les données pour qu’elles soient compréhensibles par l’API de scikit-learn. 

- On a supprimé la variable "CodeDecision" car cette information est acquise post-transaction, et la variable "ZINBIN" qui est l’identifiant des clients. Donc on a gardé 21 variables dans le modèle au final.

- On a supprimé les lignes dupliquées.

- On a remplacé les virgules par un espace. 

- On a transformé toutes les variables catégorielles en numériques (float).

- On a extrait la colonne date pour pouvoir partitionner en apprentissage et test:

	- Apprentissage : transactions ayant eu lieu entre le "2017-02-01" et le "2017- 08-31". 
  
	- Test : transactions ayant eu lieu entre le "2017-09-01" et le "2017-11-30".

### 4.1 Les différents algorithmes testés sous H20

- H2ORandomForestEstimator. 

- H2ODeepLearningEstimator.  

- AutoML 
Grâce au modèle AutoML, on a obtenu le meilleur modèle XGBOOST et pour la suite on a cherché à optimiser les paramètres pour ce modèle. 

- XGBoost  
Le dernier modèle que nous avons testé est le XGBoost. Sous H2O, nous utilisons la classe H2OXGBoostEstimator et les hyper-paramètres de l’algorithme (max_depth et ntrees) sont optimisés par GridSearchCV. On récupère le meilleur modèle avec les meilleurs paramètres et on effectue la prédiction sur les données test. 

### 4.2 Résultats :  

<img width="820" alt="Capture d’écran 2023-02-03 à 11 16 34" src="https://user-images.githubusercontent.com/114235978/216574677-ff236582-c94a-45e3-9649-07159265790c.png">
 
Pour le déploiement, on enregistre le modèle qu’il a le F1_score élevée et le taux d'erreur sur la classe positive moins élevé, c'est le modèle XGBoost avec les paramètres optimaux (ntrees, max_depth) = (50,10) (f1_score = 0.846, le taux d'erreur sur la classe positive = 0.089).

## 5 Conclusion :  

Cette étude avait pour objectif de déterminer, quels sont les algorithmes de classification les plus efficaces pour prédire la présence/absence des transactions frauduleuses en fonction des caractéristiques des transactions.

Au moyen des différentes mesures de performances de classification et des différents algorithmes mis en place pour l’ensemble de données, l’algorithme XGBoost avec l’optimisation de ses paramètres avec la méthode GRIDSERACHCV apparait le plus performant pour prédire la détection ou non des fraudes pour le jeu de données avec les techniques de rééchantillonnages comme prétraitement utilisé. Nous avons donc sélectionné et déployé ce modèle nommé « Grid_XGBoost_model ».
