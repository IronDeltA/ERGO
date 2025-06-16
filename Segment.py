import numpy as np
import motoreducteur
import joint_mot

# Définition des variables utiles


class Segment:
    # Classe définissant le comportement d'un segment de bras robot
    def __init__(self, j=0, a=0, b=0, t=0, v=np.array([])):
        # joint utilisé dans le segment
        if j!=0:
            self.seg_joint = j
        else:
            self.seg_joint = joint_mot.Joint()    

        # motoréducteur utilisé dans le segment
        self.motoreducteur = motoreducteur.Motoreducteur() 

        # matrice d'inertie du segment
        self.inertie = np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0]])

        # longueur du vecteur
        if v.any():
            self.vect = v
        else:    
            self.vect = np.array([0.0, 0.0, 0.0])

        # poids propre du segment
        self.poids_propre = 0.0

        # poids du segment
        self.poids = self.seg_joint.poids + self.poids_propre

        # centre d'inertie du segment
        self.centre_inert = np.array([[0, 0, 0]])

        self.Pose = self.seg_joint.pose
        self.update_pose(a, b, t, v=self.vect)

    def get_pose(self):
        return self.Pose

    def get_inertie(self):
        return self.inertie

    def get_poids(self):
        return self.poids

    def get_centre_inert(self):
        return self.centre_inert

    def get_vect(self):
        return self.vect

    def set_vect(self, v):
        self.vect = v
        for i in range(0, 3):
            self.Pose[i][3] = v[i]         

    def update_pose(self, a=0, b=0, t=0, v=np.array([0, 0, 0])):
        self.seg_joint.update_pose(a, b, t)
        self.set_vect(v)

    def update_rot(self, a, b, c):
        self.seg_joint.update_pose(a, b, c)
