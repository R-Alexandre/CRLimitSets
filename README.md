# limitset

Le fonctionnement est le suivant.
`./limitset m006` calcule les ensembles limites des représentations
(conjuguées à une représentation) dans PU(2,1) du groupe fondamental
de la variété m006.
La commande `./limitset m006 2` permet de ne calculer que la deuxième
représentation.

Avec `./limitset all` on calcule tous les ensembles limites du census et avec
`./limitset fractals` on calcule tous les ensembles limites qui donnent une fractale (connue), en évitant de calculer des représentations qui ne diffèrent que d'une action de Galois.

# Requis

`python` et `UNIX` sont requis.

Les librairies `snappy` et `numpy` sont requises.
Les autres sont classiques d'une
installation de python. Par ailleurs, avec la version actuelle, seul `python2`
est utilisable en raison d'un bug de snappy avec `python3`.

# Fonctionnement général

Le script se décompose en trois phases. Tous les paramètres décrits se
situent dans le fichier `parameters.py`.

## Première phase

La première phase consiste à itérer les mots de longueurs
`LENGTH_WORDS`. Ces mots sont construits en essayant au maximum d'éviter
les parties triviales (prescrites par la présentation du groupe
fondamental). Par le théorème de la trace de Goldman, on vérifie
également que les matrices correspondantes sont loxodromiques.

Chaque matrice est appliquée sur un point de l'espace projectif jusqu'à
convergence à `EPSILON` près.

Les points obtenus sont inscrits dans le fichier `points`, puis sont
triés par la commande UNIX `sort`.

## Deuxième phase

Si le nombre de points obtenus est insuffisant par rapport à `NUMBER_POINTS`,
alors la deuxième phase intervient.

La deuxième phase consiste à récupérer les points obtenus précédemment
et à itérer sur eux des mots de longueur `LENGTH_WORDS_ENRICHMENT`. Ce paramètre est automatiquement estimé (plus ou moins bien) en comparant le paramètre `NUMBER_POINTS` avec le nombre de points obtenus jusqu'alors.

Ces points sont ensuite inscrits dans le fichier `enrich`. Ce fichier est ensuite trié par la commande UNIX `sort`.

## Troisième phase

La troisième phase consiste à récupérer les derniers points obtenus et à
effectuer une bonne projection 3D.

Pour cela, on commence tout d'abord par évaluer un bon produit hermitien
correspondant à la structure CR. C'est fait par la résolution d'un système linéaire imposant que les points aient une valeur nulle par le produit hermitien, et un autre point pris comme différence du premier et dernier ait une valeur égale à -1. Le système linéaire est résolu avec une méthode des moindres carrés avec les points inscrits dans `filtered`.

Ensuite, on procède à une projection stéréographique (une projection de
Siegel est également disponible). Le point base depuis lequel on
effectue la projection
est donné par `BASE_POINT_PROJECTION`.

Les points résultants sont inscrits dans le fichier `show-stereo` si la
projection stéréographique a été effectuée, `show` sinon.
Le fichier est trié.

Si `DO_GNU_PLOT = True` alors avec le script `script-gnupic.plg`, une image est produite avec *gnuplot*.

# Résultats

Les fichiers sont
compressés avec `gzip`.

Les résultats sont placés dans le dossier `results/`.

Une fois le programme exécuté, celui-ci produit un certain nombre de fichiers
résultants. Tous sont placés initialement dans un sous-dossier de la forme
`manifolds/m006/2-6`
où `2-6` indique qu'il s'agit de la deuxième représentation et d'un calcul
effectué avec 6 lettres par mot (à la première étape).
Mais les résultats peuvent aussi être trouvés dans deux dossiers distincts de
`manifolds`.

- `pics/` recueille toutes les images fournies à la dernière étape. À noté que
dernière image calculée pour une même représentation est servie.
- `show/` recueille tous les derniers fichiers de points obtenus à la dernière étape (après projection). De
même que `pics`, seul le dernier fichié calculé pour une même représentation
paraît. Ces résultats peuvent être utilisés par des logiciels de visualisation
comme `ParaView` pour afficher les points (ce dernier s'utilise avec le
`ParticlesReader`).

# Certification

Quelques tests supplémentaires lors des calculs ont pour vocation de donner
une certification aux résultats obtenus. Ces tests sont optionnels et contrôlés
par la valeur booléenne du paramètre `CERTIFICATION`.

Lors de la première étape, les calculs sont essentiellement linéaires puisque l'on
applique une matrice sur un vecteur de C^3. Le nombre d'itérations est contrôlé
par `ITERATIONS_NUMBER`. La certification
permet de s'assurer qu'à chaque étape, les valeurs obtenus dans C^3 sont
à coordonnées en valeur absolue plus petites que `GLOBAL_PRECISION`.

Un contrôle plus décisif se fait en métrisant l'étape où l'on repasse en carte
z=1 par une inversion par z. À chaque étape, si la norme de z^{-1} est plus
grande que 1, on multiplie une constante (qui traverse toutes les itérations, et initialement vaut 1)
par cette norme. On contrôle que la constante est toujours inférieure à la
valeur de `ACCUMULATED_INVERSE_PRECISION`.

Lors de la deuxième étape, les points sont transformés une seule fois par une
matrice. Pour chaque transformation, on contrôle à nouveau que l'image est
encore à coordonnées plus petites que `GLOBAL_PRECISION`. Cette fois-ci,
on demande à ce que la norme de z et de son inverse sont toutes deux plus
petites que `ENRICH_PRECISION`.

# À faire

- Il faudrait pouvoir contrôler l'approximation des matrices pour avoir une
certification plus fiable. Actuellement, elles sont calculées par pari avec 1000 décimales de calcul (mais sans pouvoir calculer combien sont exactes, même si ça converge avec le nombre de décimales grandissant).
- Trouver plus finement et automatiquement un bon pôle de projection
stéréographique.
- Il faudrait pouvoir réduire le nombre de représentations quand le degré
(donné par `M.ptolemy_variety(3,0).degree_to_shapes()`)
est plus grand que 1. De même on devrait pouvoir se passer de calculer des représentations qui diffèrent d'une action de Galois.
