# Projet Statistiques Bayesiennes

## Introduction

L'objet de ce projet est de grouper au sein de différents clusters un ensemble de séries temporelles, et ce faisant d'estimer le modèle statistique décrivant les séries temporelles de chaque cluster.  
Pour conduire nos estimations, nous nous plaçons dans le cadre bayésien et utilisons des méthodes de simulation de Monte Carlo et chaîne de Markov cachées.  

Soit ($y_{i,t}$), avec $t=1,...,T$, une série temporelle multiple, parmi N autres series temporelles $i=1,...,N$, on fait l'hypothèse que ces séries appartiennent à K clusters différents, et que toutes les séries temporelles appartenant à un cluster $k$ sont décrites par le même modèle statistique avec un paramètre spécifique au groupe, $\theta_k$. L'appartenance à un groupe $k$ pour chaque série temporelle $i$ est inconnue à priori et est estimée en même temps que les paramètres $\theta_k$. On fait de plus l'hypothèse que les paramètres $\theta_k$ sont propres à chaque cluster.   

On introduit le vecteur $S = (S_1,S_2,...,S_N)$ où $\forall i \in 1,..,N$, $S_i$ $\in 1,...,K$ indique le groupe auquel appartient la série temprelle $i$, ainsi que le vecteur $\phi = (\eta_1, ..., \eta_K)$ où $\eta_k$ indique la proportion de série temporelle appartenant au cluster $k$.  

En utilisant un algorithme de MCMC, nous allons itérativement estimer le vecteur $S$, puis les vecteurs $\theta$ et $\phi$.

## Le modèle
### Point de vue théorique

Pour $i = 1,..,N$, la densité de $y_i$ conditionnellement à $\theta_{S_i}$ s'écrit :  

$p(y_i | \theta_{S_i})$} = \prod_{t=1}^{T} $p(y_{i,t}|y_{i,t-1},..,y_{i,0},\theta_{S_i})$

Où $p(y_i|y_{i,t-1},..,y_{i,0},\theta_{S_i})$ est une densité connue qui dépendra du modèle choisi.   

Par conséquent,  

$p(y_i | S_i, \theta_1,...,\theta_K) =  p(y_i | \theta_{S_i})$ = \left\{\begin{array}{ll}
  $p(y_i | \theta_{1})$   & \mbox{si } S_i = 1  \\
  ...                                         \\
  $p(y_i | \theta_{K})$   & \mbox{si } S_i = K
\end{array}\right.

Ensuite, on détermine un modèle probabiliste pour la variable $S = (S_1,..,S_N)$. On fait l'hypothèse que les $S_1, S_2,..,S_N$ sont deux à deux à prori indépendants et pour tout $i = 1,..,N$ on définit la probabilité à priori $Pr(S_i = k)$, la probabilité que la série temporelle $i$ appartienne au cluster $k$. On fait l'hypothèse que pour toute série $i$, on n'a à priori aucune idée d'à quel cluster elle appartient. Dès lors,
$Pr(S_i = k | \eta_1,..,\eta_K) = \eta_k$
La taille des groupes $(\eta_1,..,\eta_K)$ est à priori inconnue et est estimée grâce aux données.

## L'algorithme de MCMC

L'estimation du vecteur de paramètres $\psi = (\theta_1,..,\theta_k,\phi,S)$ à l'aide des MCMC se fait en deux étapes :
\newline
\vspace{0.4cm} 
\newline
\underline{\textbf{Etape 1}} : 
\newline 
On fixe les paramètres \textit{$(\theta_1,..,\theta_K,\phi)$} et on estime S 
\newline
Dans cette étape, on va attribuer à chaque série temporelle $i$ un groupe $k$ en utilisant la posteriori $p(S_i|y,\theta_1,..,\theta_K,\phi)$.
\newline
Par la formule de Bayes et ce qui précède, on sait que :
$p(S_i = k|y,\theta_1,..,\theta_K,\phi) \propto p(y_i|\theta_k)Pr(S_i = k|\phi)$
En utilisant les équations (1) et (3), on va calculer cette à posteriori pour $k = 1,..,K$, et à l'aide de Python, nous allons simuler un tirage de $S_i$ et lui attribuer un groupe $k$.

\vspace{0.4cm} 

\underline{\textbf{Etape 2}} : 
 
On fixe la classification $S$ et on estime le vecteur de paramètres $(\theta_1,..,\theta_K,\phi)$

Conditionnellement à $S$, les variables $\theta$ et $\phi$ sont indépendantes. Etant donné que le paramètre $\theta_k$ est propre au cluster $k$, on regroupe toutes les séries temporelles appartenant au groupe $k$.

Ainsi, $\theta_k$ est estimé en utilisant la posteriori (5) et un algorithme de Metropolis-Hastings:

$p(\theta_k|y,S_1,..,S_N) = \prod_{i : S_i = k} p(\theta_k|y_i) =  \prod_{i : S_i = k} p(y_i|\theta_k)p(\theta_k)$

Où la priori $p(\theta_k)$ dépendra du modèle choisi.

Enfin, on estime $\phi = (\eta_1,..,\eta_k$) en utilisant la posteriori (6) et un algorithme de Metropolis-Hastings : 
$p(\phi|S,y) = p(y|S,\phi,\theta_1,..,\theta_K) = p(y|S,\phi,\theta_1,..,\theta_K) \times p(S|\phi) \times p(\phi)$

$= \prod_{k=1}^{K} \prod_{i : S_i = k} p(y_i|\theta_k) \prod_{j = 1}^{N}Pr(S_j|\phi)p(\phi)$
Où la loi à priori de $\phi$ est une loi de Dirichlet(4,..,4)

On va donc estimer $\psi = (\theta_1,..,\theta_k,\phi,S)$ en répétant P fois ces deux étapes, après avoir initaliser $\psi^{0} = (\theta_1^{0},..,\theta_k^{0},\phi^{0},S^{0})$

### Implementation

D'un point de vue pratique, la vraisemblance des modèles ARIMAX(p, 0, d) estimés a été calculée à l'aide de la librairie statsmodels. 

Afin d'éviter une classification finale avec un nombre nul de séries temporelles dans un cluster, nous avons décidé de sélectionner dans ce cas une dizaine de séries de manière aléatoire, afin de pouvoir tout de même actualiser les paramètres du cluster. Dans le cas contraire, si au cours des premières itérations la taille d'un cluster est amenée à zéro, les coefficients associés au modèle ne serait alors pas actualisés.

Les deux étapes décrites dans la partie précédente ont été utilisées pour un algorithme de Gibbs. Pour chacune des étapes, un algorithme de Metropolis Hasting a été implémenté. Une marche aléatoire a été mise en place afin de trouver les coefficients des modèles, avec pour chaque étape dix itérations sucessives. Nous avons sélectionné ce nombre à la lumière des résultats que nous obtenions, ainsi qu'en prenant en compte la complexité de l'algorithme finale. 
