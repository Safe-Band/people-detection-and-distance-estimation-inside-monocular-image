# Distance estimation in moncular crowded image

Nous nous sommes intéressés à l’estimation des distances entre les personnes dans une foule à partir d’une image monoculaire. Notre solution se concentre spécifiquement sur les distances entre les têtes, qui sont les seuls éléments clairement détectables dans ce contexte. Face à l’absence de modèles directement adaptés à cette tâche dans l’état de l’art, nous avons développé une approche hybride. Celle-ci repose sur deux modèles : un modèle de détection d’objets, affiné sur un jeu de données adapté pour identifier les têtes, et un modèle d’estimation de carte de profondeur. En combinant ces deux estimations, nous avons modélisé le fonctionnement de la caméra et utilisé les lois de l’optique géométrique pour, d’une part, corriger la carte de profondeur estimée, et d’autre part, calculer les distances entre les têtes. Nos résultats sont qualitativement prometteurs, mais une évaluation quantitative reste nécessaire pour les valider pleinement.

## utilisation

Télechargez les 2 datasets suivants:
- jhu-crowd++: http://www.crowd-counting.com/
- crowdhuman : https://www.crowdhuman.org

changez le chemin d'accès de la db jhu-crowd++ in `jhu_handler.py` par celui correspondant sur votre système

la solution est présente dans le jupiter notebook `main.ipynb`, vous devrez télecharger quelques bibliothèques classiques et vous pouvez ensuite tester la solution.
Le rapport de présentation du projet est présent sous le nom `main.pdf`.
