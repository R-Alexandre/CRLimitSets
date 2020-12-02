# Requirements

The code currently works with `Python 3.8`. The required packages are described in `package_requirements.txt` and this file can be parsed to `pip`.


The program also requires `gnuplot` and `imagemagick`.

# Functioning

Set the root to `repository/simulations/`. And assume that `python` corresponds to `Python 3.8`.

- The main parameters are to be prescribed in the file `python/parameters`.
- For the simulation of the PU(2,1) representations of the figure eight-knot do `python python/eight_knot.py`. The range of the parameters is described in the same file.
- For the simulation of the unipotent representations in PU(2,1), do `./unipotent` and follow instructions.
- For the simulation of a complex hyperbolic triangle group, do `python python/triangles.py`. The parameters are described in the same file.



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

## Deuxième phase

Si le nombre de points obtenus est insuffisant par rapport à `NUMBER_POINTS`,
alors la deuxième phase intervient.

La deuxième phase consiste à récupérer les points obtenus précédemment
et à itérer sur eux des mots de longueur `LENGTH_WORDS_ENRICHMENT`. Ce paramètre est automatiquement estimé (plus ou moins bien) en comparant le paramètre `NUMBER_POINTS` avec le nombre de points obtenus jusqu'alors.

## Troisième phase

La troisième phase consiste à récupérer les derniers points obtenus et à
effectuer une bonne projection 3D.

Pour cela, on commence tout d'abord par évaluer un bon produit hermitien
correspondant à la structure CR. C'est fait par la résolution d'un système linéaire imposant que les points aient une valeur nulle par le produit hermitien, et un autre point pris comme différence du premier et dernier ait une valeur égale à -1. Le système linéaire est résolu avec une méthode des moindres carrés.

Ensuite, on procède à une projection stéréographique (une projection de
Siegel est également disponible). Le point base depuis lequel on
effectue la projection
est donné par `BASE_POINT_PROJECTION`.

Si `DO_GNU_PLOT = True` alors avec le script `script-gnupic.plg`, une image est produite avec *gnuplot*.

# Résultats

Les résultats sont placés dans le dossier `results/`.


- `pics/` recueille toutes les images fournies à la dernière étape. À noter que la
dernière image calculée pour une même représentation est servie.
- `show/` recueille tous les derniers fichiers de points obtenus à la dernière étape (après projection). De
même que `pics`, seul le dernier fichier calculé pour une même représentation
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

Un contrôle plus décisif se fait en maîtrisant l'étape où l'on repasse en carte
z=1 par une inversion par z. À chaque étape, si la norme de z^{-1} est plus
grande que 1, on multiplie une constante (qui traverse toutes les itérations, et initialement vaut 1)
par cette norme. On contrôle que la constante est toujours inférieure à la
valeur de `ACCUMULATED_INVERSE_PRECISION`.

Lors de la deuxième étape, les points sont transformés une seule fois par une
matrice. Pour chaque transformation, on contrôle à nouveau que l'image est
encore à coordonnées plus petites que `GLOBAL_PRECISION`. Cette fois-ci,
on demande à ce que la norme de z et de son inverse sont toutes deux plus
petites que `ENRICH_PRECISION`.
