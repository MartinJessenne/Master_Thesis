---
type: # Pillar | Inquiry | Logic | Trial | Evidence | Library
status: Open
related_pillar: "[[Thesis_Master_Plan]]"
tags: [thesis]
created: 2026-05-26 16:23
---
# Transcription Dernier Retour Julien

> [!info] Quick Summary
> What is this note about?

Voici la retranscription complète et ordonnée de l'entretien entre **Julien** (le superviseur) et **Martin** (l'étudiant). 

L'analyse des correspondances de voix montre un décalage initial d'environ **5 minutes et 32 secondes** entre les deux enregistrements. La piste de Julien commence plus tôt dans la phase de salutation et d'installation, tandis que la piste de Martin commence directement au moment où l'explication technique débute.

Dans la retranscription ci-dessous, les deux timestamps sont indiqués au format `[Martin / Julien]` pour vous permettre de naviguer facilement dans l'une ou l'autre piste.

---

### **Phase d'introduction et d'installation (uniquement sur la piste de Julien)**

* **Julien [03:20]** : Oui, bonjour.
* **Julien [03:22]** : Ouais, c'est bon, ça marche bien.
* **Julien [03:27]** : Ouais, ça va et vous ?
* **Julien [03:43]** : D'accord.
* **Julien [03:57]** : Ouais, dites-moi.
* **Julien [04:20]** : Ah, oui, c'est bon.
* **Julien [04:24]** : Oui, oui, c'est bon, je vois, je vois.
* **Julien [04:26]** : Oui, je vous entends. Oui, oui, c'est bon.

---

### **Début de la discussion technique**

* **Martin [00:03 / 05:35]** : Ouais, c'est ça. Et donc du coup, là on remarque à nouveau que la médiane a tendance à baisser plus on... plus les perturbations sont grandes en norme infini.
* **Julien [05:32 / 00:00]** : Humm. Ouais, d'accord, ça c'est ce dont on avait parlé la dernière fois, ouais.

* **Martin [00:19 / 05:51]** : Et aussi, donc du coup, j'ai fait plutôt le graphique ici. Donc par *mean plot*, je savais pas trop si on parlait de *mean* au sens de moyenne ou au sens de minimum, mais c'est bien au sens de moyenne, non ?
* **Julien [06:01 / 00:29]** : Ah oui, oui, bien sûr. Oui, oui, non, c'est au sens de moyenne. Oui, oui, tout à fait.
* **Martin [00:33 / 06:05]** : Ouais. OK.
* **Julien [06:07 / 00:35]** : D'accord, donc là c'est voilà... c'est plus facile... Bon, évidemment il y a des grands intervalles de confiance, ça c'est tout à fait normal. On s'attend pas vraiment à autre chose. Par contre, voilà, on voit que ça, ça diminue.

* **Martin [00:45 / 06:17]** : Et aussi, comment dire... c'est... comment... j'essaie de définir ça comme étant des... des explorations uniformes dans le sens où chaque tranche a le même nombre de... d'exploration, enfin d'expériences qui est menée dans cette tranche, de manière égale. Mais par contre c'est pas... c'est pas homogène dans le sens où, bah, les dernières tranches ont plus de volume, mais j'ai mis le même nombre de... de... je teste le même nombre de perturbations que dans les tranches de plus faible volume. Je sais pas si c'est clair.
* **Julien [06:51 / 01:19]** : Ouais, dites-moi.
* **Martin [01:17 / 06:49]** : Hum, OK.
* **Martin [01:27 / 06:59]** : OK.

* **Martin [01:27 / 06:59]** : Euh, oui, c'est quelque chose qu'il faut que je précise dans l'implémentation, mais... et ça... Et donc ça aussi, j'ai l'ai... j'ai aussi fait une autre variation de cette expérience-là... que j'ai appelée *scattered exploration*, dans le sens où, cette fois-ci, je perturbe aléatoirement avec des coefficients qui sont dans... dans $[-1, 1]$... et je calcule a posteriori, du coup, une fois que j'ai perturbé, je... j'enregistre la norme de ma matrice de perturbation et je compare avec le *duality gap*, enfin, avec la métrique que je calcule sur le total du *duality gap*. Et on trouve ce genre de *scattered plot*, quoi. Et donc on remarque que... donc plus on s'éloigne de la delta, et en effet, plus il y a de... il y a d'itérations qui convergent, quoi.
* **Julien [07:31 / 01:59]** : Ouais, c'est une question... parce que là, les... il y a quand même des points à... alors effectivement on a l'impression qu'il y a beaucoup de points à 0 lorsque vous augmentez la perturbation, à 0 ou $10$ moins quelque chose. Peut-être que ça vaut le coup d'être en log là pour l'axe... l'axe Y. Mais il y a aussi des points où ça reste assez haut finalement, donc... mais on dirait que le phénomène en moyenne effectivement tend à disparaître, ouais.
* **Martin [02:24 / 07:56]** : À delta, ah, c'est une bonne question. J'ai... j'avoue que j'ai pas... On en avait parlé, j'avais dit qu'il fallait que je le fasse pour une...
* **Martin [02:30 / 08:02]** : Rajouter une perturbation... où la valeur de la norme de perturbation est de 0, comme ça on aurait pu avoir la valeur de la delta. Mais je l'ai pas fait. Ça, il faut que je le note.
* **Julien [08:39 / 03:07]** : Ouais.
* **Julien [08:42 / 03:10]** : Ouais, mais en fait, on pourrait aussi se demander... enfin, vous voyez les... les points, je suppose que à delta c'est... le point le... est-ce que c'est le point le plus à gauche ?
* **Martin [02:44 / 08:16]** : Hum.
* **Martin [02:47 / 08:19]** : Ouais.
* **Martin [02:51 / 08:23]** : Ouais.
* **Martin [03:00 / 08:32]** : Hum.
* **Martin [03:09 / 08:41]** : Non, là c'est le...
* **Martin [03:11 / 08:43]** : Ouais... justement. Là, attendez, j'essaie de voir... Hum... *scatter* métrique actuel...

* **Martin [03:28 / 09:00]** : Euh, là c'est le... maximum, oui c'est ça, là c'est le... la valeur, c'est le maximum des 10 %... des 10 % des dernières valeurs. Mais je peux faire ici... changer et mettre *total var*. Hop. Mince.
* **Martin [03:44 / 09:16]** : Et je peux relancer.
* **Julien [09:15 / 03:43]** : D'accord.
* **Martin [03:48 / 09:20]** : Exactement, ouais, exactement, ouais, exactement. Et donc là, du coup, on est dans... là c'est les... c'est *total variation*.
* **Martin [04:00 / 09:32]** : Donc, sur l'axe des ordonnées, du coup, on voit que c'est des valeurs qui sont un peu plus grandes... mais globalement le... le profil est à peu près similaire.
* **Martin [04:09 / 09:41]** : Là, ici, c'est pareil, ça c'est... ça c'est pour le... les explorations concentriques, et c'est aussi des *total variation*.
* **Julien [09:43 / 04:11]** : Ouais, dites-moi.

* **Martin [04:21 / 09:53]** : Et pour rester aussi sur la *total variation*, du coup, c'est... je peux faire la transition sur la deuxième chose que j'ai implémentée, qui était... la sorte de *cumulative total variation* dont on parlait. Pour avoir le... le... comment dire... La *total variation*, ça dit un peu, sur un historique des *duality gap*, c'est une métrique qui est réduite et qui dit à quel point il y a eu des... des variations abruptes d'une itération à l'autre. Et ce qu'on voulait, c'était plutôt... c'était... ne pas développer cette... enfin, développer cette métrique-là, pour savoir au fil du temps, cumulativement, à quel point est-ce que les écarts sont de plus en plus grands ou pas. Je sais pas si c'est clair, je sais pas si j'ai bien... j'ai bien mis le... le processus.
* **Julien [10:40 / 05:08]** : Attendez, re- redites-moi ça en montrant la figure peut-être, pour que je voie, euh...

* **Martin [05:10 / 10:42]** : Ouais, là justement. En fait, là ce que je veux vous montrer, c'est... c'est le profil... c'est il faut considérer ça comme une seule expérience, donc... Vous voyez, ça c'est mon... c'est le... je sais pas si vous voyez en direct à l'écran le graphe, ça représente donc, là, la... la croix rouge c'est le... le point de départ de ma stratégie, donc je l'initialise au... au centre du simplexe, à 0.5, 0.5. Et on voit, du coup, pour OMW, du coup, le... la manière dont les... dont les stratégies s'updatent et... comme quoi elle gravite autour du... *Nash equilibrium*, qui est lui situé à... ici, donc... pour X c'est 0.1, 0.9, et pour Y c'est à peu près 0.2, 0.8, quoi.
* **Julien [11:00 / 05:28]** : Ouais, ouais, je vois, ouais.
* **Martin [05:53 / 11:25]** : Et donc, là sur une... sur un seul... sur un seul jeu, pour une seule matrice, pour une seule matrice, du coup, on a l'historique des *duality gap* qui est ici.
* **Julien [11:24 / 05:52]** : Ouais, ouais, d'accord.
* **Julien [11:40 / 06:08]** : Attendez, là, ce que vous faites, c'est que vous... vous... on a l'équilibre de Nash dans le simplexe, on peut choisir où est-ce qu'il est, et vous le... vous le faites se déplacer, d'accord. Et qu'est-ce que vous... qu'est-ce que vous suivez comme métrique là-dessus ?

* **Martin [06:13 / 11:45]** : Non, ça c'est... justement, ça c'est mes... ça c'est les matrices paramétriques, ça. Pour que je puisse avoir justement ce... placer le *Nash equilibrium* où je veux dans le... dans le simplexe.
* **Julien [11:59 / 06:27]** : Mmh. Ouais, d'accord.
* **Martin [06:29 / 12:01]** : Ouais, exactement. Justement, ça aussi, il faudrait qu'on en parle, parce que j'ai essayé de... de creuser un peu sur comment j'avais trouvé cette matrice-là, parce que j'avais trop de degrés de liberté, et donc c'est encore un truc dont on pourra parler après. Euh, mais du coup globalement, oui, c'est ça, c'est la matrice avec des équilibres de Nash para- que je peux paramétrer dans les coefficients de la matrice, quoi. Et là, en particulier, je les ai paramétrés pour que... l'équilibre de Nash se place ici dans le... dans le simplexe.
* **Martin [06:54 / 12:26]** : Et donc on voit que le profil stratégique orbite autour. Ça c'est le *duality gap* au cours du temps.
* **Julien [12:12 / 06:40]** : Ouais, c'est très bien, en fait, euh...
* **Martin [07:03 / 12:35]** : Et... et ici, en bas, du coup, justement, j'ai ploté... donc c'est un graphique en log... les... la valeur cumulative de... de la *total variation*.
* **Julien [12:35 / 07:03]** : Oui, oui, c'est ça, c'est la version de...
* **Julien [12:39 / 07:07]** : D'accord, d'accord.
* **Martin [07:24 / 12:56]** : Euh, bah je pense que le... comment dire... le graphe en log est peut-être pas... peut-être pas très adapté pour bien comprendre le... pour bien saisir le... pour bien analyser, plutôt, ce profil-là. Mais... ouais.
* **Julien [12:56 / 07:24]** : Ah oui, d'accord. Oui, oui, d'accord, avec... d'accord. Donc là, là vous paramétrez, c'est le Delta X, Delta Y ou vous c'est les lambda...
* **Martin [07:43 / 13:15]** : Ouais.
* **Martin [07:49 / 13:21]** : Hum.
* **Julien [13:13 / 07:41]** : Mmh, ouais.
* **Julien [13:21 / 07:49]** : D'accord, ouais.

* **Martin [08:11 / 13:43]** : Mmh, c'est une très bonne échelle.
* **Julien [14:02 / 08:30]** : Ah, c'est très bien. En fait, euh...
* **Martin [08:24 / 13:56]** : Ouais.
* **Julien [14:05 / 08:33]** : Voilà, ça c'est ce qu'on attend, et...
* **Martin [08:30 / 14:02]** : Ouais.
* **Martin [08:34 / 14:06]** : Ouais, je vais commencer.
* **Martin [08:40 / 14:12]** : Voilà.
* **Julien [14:17 / 08:45]** : Attendez, remontez dans vos graphes, là, qui regardaient quand est-ce que X est proche de la... de l'équilibre de Nash. Voilà, là, c'est très bien, on voit que c'est loin.
* **Martin [08:58 / 14:30]** : Mmh.
* **Julien [14:31 / 08:59]** : Théoriquement, il est là.

* **Martin [08:59 / 14:31]** : Et donc aussi, un peu une des dernières choses sur laquelle je me suis concentré et que je vais continuer à creuser, c'est... parce qu'en fait je pense que j'avais un peu mal interprété les choses au début de mon travail, c'est-à-dire que je considérais... vous savez, j'avais essayé de paramétrer les jeux de matrices... avec des valeurs du Nash équilibre... du Nash équilibre qui... qui pour les valeurs de X, suivaient un profil quadratique, et Y linéaire. C'est un peu arbitraire comme approche. Et... je voyais pas ça d'un milieu géométrique, mais plutôt d'une manière... comment dire... analytique ou aussi, comment dire... sur le... enfin si, quand même toujours un peu géométrique, mais c'est-à-dire que si le long d'une coordonnée on tendait plus vite que le long de l'autre, on allait avoir des comportements plus chaotiques pour OMW que... que dans d'autres cas normaux. Enfin, je sais pas si c'est très clair. Mais, du coup, le principe maintenant, c'est que je vois ça plus d'une manière géométrique, et donc là, en fait, ce que je fais, c'est que je crée des courbes paramétriques ici, tout au début.
* **Julien [14:50 / 09:18]** : Ouais.
* **Julien [14:52 / 09:20]** : Mmh, ouais.
* **Julien [14:55 / 09:23]** : Ça c'est après un certain nombre d'itérations.
* **Julien [15:01 / 09:29]** : Voilà.
* **Julien [15:11 / 09:39]** : Mmh, ouais.
* **Julien [15:18 / 09:46]** : Ouais, d'accord.

* **Martin [09:55 / 15:27]** : Donc par exemple, là, actuellement, ce que j'ai fait, c'est que j'ai fait un cercle. Voilà, donc je peux faire soit un... un... des courbes paramétriques linéaires ou circulaires.
* **Julien [15:25 / 09:53]** : Ah, ben c'est pas à delta du coup, là, ici.
* **Martin [10:16 / 15:48]** : Et en gros, le principe c'est de déplacer l'équilibre de Nash, par exemple, je peux vous montrer pour... ce qui est assez intéressant, c'est pour 0.3...
* **Julien [15:28 / 09:56]** : Centré, bah... euh... bah... c'est-à-dire que...
* **Martin [10:23 / 15:55]** : Ouais.
* **Julien [15:39 / 16:07]** : Bah, le jeu c'est... c'est un jeu paramétrique dont le l'équilibre de Nash dépend de lambda et gamma, et lambda et gamma suivent un cercle en quelque sorte. Donc c'est-à-dire que du coup...

* **Martin [10:30 / 16:02]** : Donc pour ODJ, du coup, on voit que... vu que pour ODJ ça repose sur une... un régulariseur qui est, comment dire... euclidien, donc si on... le... comment dire... l'équilibre de Nash se déplace le long d'un... d'un cercle dans le... comment dire... dans le... dans le... dans le simplexe...
* **Julien [15:53 / 10:21]** : Ah, centré sur 0.5, 0.5, le centre du simplexe.
* **Martin [10:52 / 16:24]** : pour le... pour ODJ, il y a pas de soucis pour converger, vu que... la métrique elle est régulière le long de ce cercle, en fait.
* **Martin [11:01 / 16:33]** : Je sais pas si c'est clair.
* **Julien [16:00 / 10:28]** : OK, euh...
* **Julien [16:03 / 10:31]** : Bah c'est, on peut le faire, de toute façon, autour de... comment dire... du... de... de l'équilibre de Nash de à delta, dans ce cas-là.

* **Martin [11:07 / 16:39]** : Je peux vous montrer sur le graphe, ouais.
* **Julien [16:12 / 10:40]** : Ouais, ouais.
* **Martin [11:09 / 16:41]** : Ouais, c'est ça, en fait je peux vous montrer, ouais.
* **Julien [16:14 / 10:42]** : Ça c'est possible, ouais.
* **Martin [11:12 / 16:44]** : Exactement.
* **Julien [16:20 / 10:48]** : Ouais.
* **Martin [11:16 / 16:48]** : Bah... par exemple, ouais, donc le truc c'est que l'initialisation elle se fait toujours au centre du simplexe, en 0.5, 0.5. Et on peut voir que là, du coup, bah, le point... l'équilibre de Nash, donc le point vert, euh, il est à... à la gauche du début, et si j'avance... hop, il se retrouve en bas, et donc je... comme ça un peu je balaye le simplexe le long d'un cercle, quoi.
* **Julien [16:20 / 10:48]** *(sur l'autre piste)* : Ouais, c'est une question...
* **Martin [11:47 / 17:19]** : Exactement. Et donc... c'est ça, et donc du coup, on remarque que... donc pour... que ODJ le long de ces... ces courbes paramétriques, a aucun problème de... de divergence ou de stabilité numérique.
* **Martin [12:02 / 17:34]** : Là c'est... ouais, c'est la toute toute dernière valeur du... des stratégies après avoir fait tourner l'algorithme d'ODJ.
* **Julien [16:38 / 11:06]** : Mmh.

* **Martin [12:17 / 17:49]** : Si, bah, juste... bah... comment dire, le duality gap c'est un... c'est ça se calcule à partir des stratégies, donc si... si les stratégies X sont très proches de celles qui sont... qui sont théoriques pour le... pour l'équilibre de Nash, on... on s'attend à ce que le duality gap soit très bas. D'ailleurs, le duality gap il est...
* **Martin [12:39 / 18:11]** : Ouais, c'est ça, elles se superposent là, c'est-à-dire que la courbe pointillée elle est superposée with la courbe pleine. La courbe pointillée elle représente le... l'équilibre de Nash tel qu'il a été paramétré dans le... le jeu de la matrice, et la courbe pleine, elle représente ce... l'équilibre de Nash qu'a trouvé l'algorithme après avoir tourné pour les itérations.
* **Julien [17:21 / 11:49]** : Ouais, oui, oui, d'accord, c'est pas le simplexe, c'est plutôt les lignes de niveau.
* **Martin [12:59 / 18:31]** : C'est l'intérieur, OK.
* **Julien [17:29 / 11:57]** : Ouais.
* **Martin [13:03 / 18:35]** : Oui.
* **Julien [17:48 / 12:16]** : Ouais, ouais, c'était ça, ouais.
* **Martin [13:08 / 18:40]** : Oui, OK, ouais, ouais, c'est clair. Hum.
* **Julien [17:51 / 12:19]** : Ouais.
* **Martin [13:22 / 18:54]** : Mmh.

* **Martin [13:26 / 18:58]** : Donc ça, on voit... comme ça on voit pour ODJ.
* **Julien [18:00 / 12:28]** : Oui, c'est ça.
* **Martin [13:32 / 19:04]** : OK, théoriquement il est là, mais c'est pas très clair, parce que...
* **Julien [18:04 / 12:32]** : Ouais.
* **Martin [13:38 / 19:10]** : Bah... en soi, on voit du coup que c'est des valeurs qui sont de l'ordre de 10^-8. Là en... là c'est le maximum des 10 %... des 10 % du duality gap.
* **Julien [18:08 / 12:36]** : Mmh.
* **Martin [13:48 / 19:20]** : Ouais.
* **Julien [18:13 / 12:41]** : OK.
* **Martin [13:52 / 19:24]** : Mmh, ouais. Ouais, j'ai l'impression que ça converge.
* **Julien [18:26 / 12:54]** : Ouais.
* **Julien [18:28 / 12:56]** : OK.

* **Martin [13:55 / 19:27]** : Et, du coup, ici, pareil, mais pour... OMW, et on voit que déjà le comportement commence un peu... à être un peu plus chaotique... pour le rayon... donc là c'est un cercle de rayon 0.3, je crois. Et plus on fait augmenter le rayon, et plus ça devient chaotique.
* **Martin [14:14 / 19:46]** : Ça, à chaque fois, c'est... c'est les... la dernière itération du... lorsqu'on fait tourner le jeu, quoi.
* **Martin [14:24 / 19:56]** : Eu, c'est 10000 je crois, ouais.
* **Martin [14:32 / 20:04]** : Ah, oui.
* **Martin [14:46 / 20:18]** : Ouais.

* **Martin [14:48 / 20:20]** : C'est ça, c'est pour OMW. En fait, théoriquement ce qu'il faut comprendre, c'est que... ça c'est le graphe global sur toutes les explorations qu'on a fait le long des courbes paramétriques, et que ensuite, avec cette... vue ici, ce qu'on fait c'est qu'on zoome sur une itération. Par exemple là, c'est la 304e... 340... 347e, et on regarde, voilà, ce qui se passe dedans, voilà.
* **Martin [15:13 / 20:45]** : Ouais.
* **Martin [15:24 / 20:56]** : Ah, bah c'est pas à delta, du coup, là, ici.
* **Martin [15:31 / 21:03]** : Centré, bah... euh... bah... c'est-à-dire que...
* **Martin [15:39 / 21:11]** : Bah, le jeu c'est... c'est un jeu paramétrique dont le l'équilibre de Nash dépend de lambda et gamma, et lambda et gamma suivent un cercle en quelque sorte. Donc c'est-à-dire que du coup...
* **Martin [15:53 / 21:25]** : Ah, centré sur 0.5, 0.5, le centre du simplexe.
* **Martin [16:00 / 21:32]** : OK, euh...
* **Martin [16:03 / 21:35]** : Bah c'est, on peut le faire, de toute façon, autour de... comment dire... du... de... de l'équilibre de Nash de à delta, dans ce cas-là.
* **Martin [16:12 / 21:44]** : Ouais, ouais.
* **Martin [16:14 / 21:46]** : Ça c'est possible, ouais.
* **Martin [16:20 / 21:52]** : Ouais.
* **Martin [16:29 / 22:01]** : Hum.
* **Martin [16:35 / 22:07]** : Ouais.

* **Martin [16:38 / 22:10]** : Ouais, je vais essayer de faire ça. Mais, sur ce, justement, j'avais une question par rapport à ça, c'est que... donc, le simplexe pour ODJ c'est... c'est assez simple, vu que c'est un... régulariseur qui est, comment dire... euclidien, c'est-à-dire que c'est juste un cercle, non ?
* **Martin [16:59 / 22:31]** : Pour les distances, non ?
* **Martin [17:03 / 22:35]** : La... la forme, la forme du simplexe... pour ODJ... Enfin, j'avais besoin... j'avais besoin un peu de... de clarifier à quoi ressemble le simplexe selon les... les régulariseurs qu'on utilise.
* **Martin [17:21 / 22:53]** : Ouais, oui, oui.
* **Martin [17:24 / 22:56]** : Oui, c'est pas le simplexe, c'est plutôt les lignes de niveau, ouais, ouais.
* **Martin [17:29 / 23:01]** : Ouais.
* **Martin [17:34 / 23:06]** : Hum, ouais, ça c'est clair.
* **Martin [17:48 / 23:20]** : Ouais, ouais, c'était ça dans... ouais.
* **Martin [17:51 / 23:23]** : Ouais.
* **Martin [17:53 / 23:25]** : Ouais, c'est ça, et donc du coup, pour la norme euclidienne, on est d'accord que c'est des... c'est des cercles, les...
* **Martin [18:00 / 23:32]** : OK.
* **Martin [18:04 / 23:36]** : Ouais.
* **Martin [18:08 / 23:40]** : Hum.
* **Martin [18:13 / 23:45]** : OK.
* **Martin [18:26 / 23:58]** : Ouais.
* **Martin [18:28 / 24:00]** : OK.

* **Martin [18:31 / 24:03]** : Et donc, justement, si je me pose la question par rapport à ça, c'est... est-ce que... parce que là du coup, si on prend, si on fait cette... cette exploration autour de... de à delta...
* **Martin [18:44 / 24:16]** : mais en faisant un cercle autour, est-ce que il y a pas des... est-ce qu'on risque pas de sortir du simplexe, justement, en faisant un cercle autour de...
* **Julien [18:59 / 13:27]** : Ouais.
* **Martin [18:59 / 24:31]** : OK.
* **Julien [19:08 / 13:36]** : OK.
* **Martin [19:08 / 24:40]** : Ouais.
* **Martin [19:12 / 24:44]** : OK.

* **Martin [19:19 / 24:51]** : Mais en fait, ça, ça, je peux même le faire là en... en live, parce que c'est pas très compliqué, il faut juste que je trouve... je me souviens plus c'était... qu'est-ce que c'était les... les centres de... enfin le la valeur du des Nash équilibre pour... pour à delta. Euh, je vais chercher. Chercher maintenant.
* **Julien [19:46 / 14:14]** : Ouais.
* **Martin [19:46 / 25:18]** : Mmh, ouais.
* **Julien [19:48 / 14:16]** : Mmh, ouais.
* **Martin [19:48 / 25:20]** : Ouais.
* **Julien [20:59 / 15:27]** : Ouais.
* **Martin [20:59 / 26:31]** : Ouais.
* **Julien [21:09 / 15:37]** : Ouais, j'ai les coefficients, c'est ça.
* **Martin [21:09 / 26:41]** : Ouais, j'ai les coefficients, c'est ça.

* **Martin [21:12 / 26:44]** : OK, je peux faire ça, donc on va faire Delta... est égal à 0.1, et donc le centre, ce serait... 1 sur... 1 plus Delta, OK, et ici, ce serait... euh, 1 sur... Et R ce serait déjà résolu, c'est ça ?
* **Julien [22:03 / 16:31]** : Ouais.
* **Martin [22:03 / 27:35]** : Ouais.
* **Martin [22:09 / 27:41]** : Ouais.
* **Julien [22:09 / 16:37]** : Ouais, presqu'un simplexe.
* **Martin [22:18 / 27:50]** : Ouais, presqu'un simplexe.
* **Martin [22:30 / 28:02]** : Ouais, je crois que ça fonctionne, là, juste Delta, qu'est-ce qu'il me... Ah, est-ce que j'ai déjà... OK, je vais faire comme ça.
* **Julien [22:30 / 16:58]** : Ouais, je crois que ça fonctionne, là, juste Delta, qu'est-ce qu'il me... Ah, est-ce que j'ai déjà... OK, je vais faire comme ça.
* **Martin [22:39 / 28:11]** : Voilà.
* **Julien [22:39 / 17:07]** : Voilà.
* **Julien [22:41 / 17:09]** : Ouais, on peut tester.
* **Martin [22:44 / 28:16]** : Euh, OK, on peut tester.
* **Martin [22:50 / 28:22]** : OK, donc ODJ, ça fonctionne bien.
* **Julien [22:50 / 17:18]** : OK, donc ODJ, ça fonctionne bien.
* **Martin [22:53 / 28:25]** : Et ouais, pour OMW, du coup, on voit que c'est un peu chaotique.
* **Julien [22:53 / 17:21]** : Et ouais, pour OMW, du coup, on voit que c'est un peu chaotique.

---

### **Phase de conclusion et d'organisation**

* **Martin [23:14 / 28:46]** : Ouais.
* **Julien [23:14 / 17:42]** : Ouais.
* **Martin [23:18 / 28:50]** : Mmh.
* **Julien [23:18 / 17:46]** : Mmh.
* **Martin [23:25 / 28:57]** : Ouais.
* **Julien [23:25 / 17:53]** : Ouais.
* **Martin [23:28 / 29:00]** : Mmh.
* **Julien [23:28 / 17:56]** : Mmh.
* **Martin [23:36 / 29:08]** : Ouais.
* **Julien [23:36 / 18:04]** : Ouais.
* **Martin [23:38 / 29:10]** : Ouais.
* **Julien [23:38 / 18:06]** : Ouais.
* **Martin [23:51 / 29:23]** : Oui, oui.
* **Julien [23:51 / 18:19]** : Oui, oui.
* **Martin [23:53 / 29:25]** : Mmh.
* **Julien [23:53 / 18:21]** : Mmh.
* **Martin [23:59 / 29:31]** : Ouais, j'essaierai de faire ça.
* **Julien [23:59 / 18:27]** : Ouais, j'essaierai de faire ça.

* **Martin [24:05 / 29:37]** : Ouais, c'est ça, j'étais en train de réfléchir à la... à la manière d'articuler tout ça, et... OK.
* **Julien [24:05 / 18:33]** : Ouais, c'est ça, j'étais en train de réfléchir à la... à la manière d'articuler tout ça, et... OK.
* **Martin [24:12 / 29:44]** : Ouais, ça marche. Je vais essayer de... de finaliser quelque chose dans les prochains jours pour que... vous puissiez relire avant de publier la dernière version, quoi.
* **Julien [24:12 / 18:40]** : Ouais, ça marche. Je vais essayer de... de finaliser quelque chose dans les prochains jours pour que... vous puissiez relire avant de publier la dernière version, quoi.
* **Martin [24:19 / 29:51]** : Ouais.
* **Julien [24:19 / 18:47]** : Ouais.
* **Martin [24:23 / 29:55]** : Bah, en fait, 1er juin, je crois que c'est la dernière version.
* **Julien [24:23 / 18:51]** : Bah, en fait, 1er juin, je crois que c'est la dernière version.
* **Martin [24:26 / 29:58]** : Et après... ouais.
* **Julien [24:26 / 18:54]** : Et après... ouais.
* **Martin [24:29 / 30:01]** : C'est pour ça, je vais essayer de me dépêcher, et après, bah il y a la soutenance... un peu plus tard.
* **Julien [24:29 / 18:57]** : C'est pour ça, je vais essayer de me dépêcher, et après, bah il y a la soutenance... un peu plus tard.

* **Martin [24:49 / 30:21]** : Ouais.
* **Julien [24:49 / 19:17]** : Ouais.
* **Martin [24:54 / 30:26]** : Ouais.
* **Julien [24:54 / 19:22]** : Ouais.
* **Martin [25:01 / 30:33]** : Mmh.
* **Julien [25:01 / 19:29]** : Mmh.
* **Martin [25:03 / 30:35]** : OK.
* **Julien [25:03 / 19:31]** : OK.

* **Martin [25:05 / 30:37]** : Et, ouais, et pour la soutenance, il faut que je fasse un PowerPoint, des slides et tout ?
* **Julien [25:05 / 19:33]** : Et, ouais, et pour la soutenance, il faut que je fasse un PowerPoint, des slides et tout ?
* **Martin [25:12 / 30:44]** : Ah, non, justement, je... enfin, j'ai pas regardé, mais...
* **Julien [25:12 / 19:40]** : Ah, non, justement, je... enfin, j'ai pas regardé, mais...
* **Martin [25:19 / 30:51]** : OK.
* **Julien [25:19 / 19:47]** : OK.
* **Martin [25:25 / 30:57]** : OK.
* **Julien [25:25 / 19:53]** : OK.
* **Martin [25:28 / 31:00]** : OK, est-ce que ça marche ?
* **Julien [25:28 / 19:56]** : OK, est-ce que ça marche ?
* **Martin [25:32 / 31:04]** : Euh, OK, ben ça fonctionne pour moi dans ce cas-là.
* **Julien [25:32 / 20:00]** : Euh, OK, ben ça fonctionne pour moi dans ce cas-là.
* **Martin [25:44 / 31:16]** : OK.
* **Julien [25:44 / 20:12]** : OK.

* **Martin [26:42 / 32:14]** : OK, ça marche. Ben, je vais essayer de m'en charger.
* **Julien [26:42 / 21:10]** : OK, ça marche. Ben, je vais essayer de m'en charger.
* **Martin [27:11 / 32:43]** : OK, ça marche. Merci, au revoir.
* **Julien [27:11 / 21:39]** : OK, ça marche. Merci, au revoir.