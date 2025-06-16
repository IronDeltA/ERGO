# ERGO
Educational Robotics for General Operations

Ce projet a pour objectif d'aboutir au développement d'un bras robotique destiné à l'usinage du bois à des fins artistiques.
Dans ce but, les étapes envisagées doivent permettre au processus d'être évolutif, général et reproductible.
 + Conception d'un outil de simulation d'architecture robotique pour robots à joints rotatifs et prismatiques, résolution des modèles cinématiques directe et inverse, modélisation statique et dynamique et planification de trajectoire
 + Conception 3D de l'architecture du bras robot et des motoréducteurs
 + Conception des cartes électroniques
 + interfaçage électronique des cartes, capteurs et actionneurs
 + Mise en place d'un contrôle ROS
 + Création d'une sourcouche logicielle à destination du travail du bois
 + Ajout d'une plateforme rotative et gestion conjointe du bras et de la plateforme pour l'usinage


Ces Milestones peuvent être amenées à évoluer au fil du projet

---------------------------------------------------------------------------------------------------------------------------

Etat d'avancement:
 + Outil de simulation en cours de rédaction. Robot.py incorpore un module encapsulant une classe Robot comprenant pour l'instant les méthodes de classe permettant la création d'un robot paramétré selon les paramètres Denavit Hartenberg. Plus de détails dans le fichier. le fichier main.py comprend une instantiation élémentaire d'un robot imitant l'architecture UR.
