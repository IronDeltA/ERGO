# ------------- Informations générales ----------------------- #

#  ## Version du 16/06/2025
#  ## Auteur: Amaury Pinéda
#  ## Projet Open Source BRAS ROBOT

#  ## Module de modélisation robotique d'un robot à nb_j joints

# ------------- Etat des fonctionnalité ---------------------- #

# Implémentation DH pour un nombre dejoints défini: terminé
# Cinématique directe globale et finale: terminé
# Calcul de la jacobienne: terminé
# GUI: en cours : pas encore la possibilité de choisir joint rota/prisma

# Cinématique inverse: prise en compte d'un poignet sphérique ou non
# et donc résolution analytique ou numérique.
# Ajout résolution analytique/numérique et possibilité de choisir (à faire)
# Ajout des éléments dynamiques pour les équations de Lagrange (à faire)
# Prise en compte des collisions, singularités, planif trajectoires (à faire)

# ------------ Optimisations à réaliser ---------------------- #

# Optimisations de l'encapsulation des fonctions cinématiques
# optimisation des formules pour le temps de calcul
# identification des éléments nuls -> remplacer dans le symbolique
# Optimisation du GUI
# Commenter le code


# ------------ CODE ------------------------------------------ #

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wd


# ------------ définition des symboles constants ------------- #

# Variables pour la vérification du poignet sphérique
delta = sp.symbols('delta1:4')

# Position du TCP dans son repère propre
x, y, z = sp.symbols('x y z')

# Variables de demande d'une position de fin
x_t, y_t, z_t = sp.symbols('xt yt zt')
euler = sp.symbols('euler1:4')

# ------------ définition de l'élément robot ----------------- #

class Robot:
    def __init__(self,
                 joints=0,
                 t_j=None,
                 theta=None,
                 alpha=None,
                 a=None,
                 d=None,
                 x=None,
                 y=None,
                 z=None):

        # définition du nombre de joints 
        if joints == 0:
            self.nb_j = 6
        else:
            self.nb_j = joints  # nombre de joints

        if t_j is None or not np.any(t_j):
            self.type_j = np.array(['R', 'R', 'R', 'R', 'R', 'R'])
        else:
            self.type_j = t_j

        # initialisation des symboles propres
        self.theta_sym = sp.symbols(f'theta1:{self.nb_j+1}')
        self.alpha_sym = sp.symbols(f'alpha1:{self.nb_j+1}')
        self.a_sym = sp.symbols(f'a1:{self.nb_j+1}')
        self.d_sym = sp.symbols(f'd1:{self.nb_j+1}')

        self.euler = sp.symbols('euler1:4')
        self.x_t, self.y_t, self.z_t = sp.symbols('xt yt zt')

        self.IK = sp.zeros(4)
        self.IK[:3, :3] = np.dot(np.dot(sp.rot_axis1(self.euler[0]), sp.rot_axis2(self.euler[1])),
                                 sp.rot_axis3(self.euler[2]))
        self.IK[:3, 3] = sp.Matrix([self.x_t, self.y_t, self.z_t])

        self.IK_n = sp.lambdify((self.euler, self.x_t, self.y_t, self.z_t), self.IK, 'numpy')

        # initialisation des chaînes articulées
        self.links, self.links_cumul = self.links_set()  # tableau matrices DH
        self.TCP_vec = sp.Matrix([x, y, z, 1])           # pos du TCP dans la base n

        # paramètres initiaux DH
        if not (theta.any()):
            self.theta = np.zeros([self.nb_j], dtype=float)
        else:
            self.theta = theta
        if not (alpha.any()):
            self.alpha = np.zeros([self.nb_j], dtype=float)
        else:
            self.alpha = alpha
        if not (a.any()):
            self.a = np.zeros([self.nb_j], dtype=float)
        else:
            self.a = a
        if not (d.any()):
            self.d = np.zeros([self.nb_j], dtype=float)
        else:
            self.d = d

        # paramètres initiaux du TCP
        if x is None:
            self.x = 0
        else:
            self.x = x
        if y is None:
            self.y = 0
        else:
            self.y = y
        if z is None:
            self.z = 0
        else:
            self.z = z

        # Equation de mouvement, fonctions de cin directe symb/num et sym cumulée
        self.DK_s, self.DK_n, self.DK_npart = self.DK_rob()
        # Jacobienne symb/num
        self.Jac_s, self.Jac_n = self.Jac_rob()

    # Formation des matrices DH (symbolique) totale et cumulées
    def links_set(self):
        T_tot = []
        T_cumul = []
        Ti_cumul = sp.eye(4)
        for i in range(0, self.nb_j):
            T1i = sp.Matrix([[sp.cos(self.theta_sym[i]),
                              -sp.sin(self.theta_sym[i]), 0.0, 0.0],
                            [sp.sin(self.theta_sym[i]),
                             sp.cos(self.theta_sym[i]), 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

            T2i = sp.Matrix([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, self.d_sym[i]],
                            [0.0, 0.0, 0.0, 1.0]])

            T3i = sp.Matrix([[1.0, 0.0, 0.0, self.a_sym[i]],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

            T4i = sp.Matrix([[1.0, 0.0, 0.0, 0.0],
                            [0.0, sp.cos(self.alpha_sym[i]),
                             -sp.sin(self.alpha_sym[i]), 0.0],
                            [0.0, sp.sin(self.alpha_sym[i]),
                             sp.cos(self.alpha_sym[i]), 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

            Ti_tot = T1i*T2i*T3i*T4i
            T_tot.append(Ti_tot)
            Ti_cumul *= Ti_tot
            T_cumul.append(Ti_cumul)
        return T_tot, T_cumul

    # getter
    def get_position(self, mat):
        return mat[0:3, 3]

    def get_theta(self, i):
        return self.theta[i]

    def get_alpha(self, i):
        return self.alpha[i]

    def get_a(self, i):
        return self.a[i]

    def get_d(self, i):
        return self.d[i]

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z
 
    # setter
    def set_theta(self, t, i=None):
        if i is None:
            self.theta = t
        else:
            self.theta[i] = t

    def set_alpha(self, a, i=0):
        if i == 0:
            self.alpha = a
        else:
            self.alpha[i] = a

    def set_a(self, a, i=0):
        if i == 0:
            self.a = a
        else:
            self.a[i] = a

    def set_d(self, d, i=0):
        if i == 0:
            self.d = d
        else:
            self.d[i] = d

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_z(self, z):
        self.z = z

    # Définition de la cinématique directe symbolique et numérique du robot
    def DK_rob(self):
        i = 1
        T_TCP = sp.Matrix.eye(4)
        T_TCP[:, 3] = self.TCP_vec
        f_part = []
        T = sp.Matrix([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])

        # cinématique numérique cumulée
        for e in self.links:
            T *= e
            if i == 1:
                f_part.append(sp.lambdify((self.theta_sym[0],
                                           self.alpha_sym[0],
                                           self.d_sym[0],
                                           self.a_sym[0]),
                              T,
                              'numpy'))

            elif i < self.nb_j:
                f_part.append(sp.lambdify((self.theta_sym[0:i],
                                           self.alpha_sym[0:i],
                                           self.d_sym[0:i],
                                           self.a_sym[0:i]),
                              T,
                              'numpy'))
            else:
                f_part.append(sp.lambdify((self.theta_sym[0:i],
                                           self.alpha_sym[0:i],
                                           self.d_sym[0:i],
                                           self.a_sym[0:i],
                                           x,
                                           y,
                                           z),

                              T*T_TCP,
                              'numpy'))
            i += 1
        # T: matrice symbolique
        T_fin = T*T_TCP

        # f: fonction numérique appelable
        f = sp.lambdify((self.theta_sym,
                         self.alpha_sym,
                         self.d_sym,
                         self.a_sym,
                         x, y, z), T_fin, 'numpy')
        return T_fin, f, f_part

    # fonction numérique basée sur la symbolique "lambdifiée"
    def DK_num(self, arg="pose"):
        if arg == "pose":
            return self.DK_n(self.theta,
                             self.alpha,
                             self.d,
                             self.a,
                             self.x,
                             self.y,
                             self.z)

        elif arg == "position":
            return self.DK_n(self.theta,
                             self.alpha,
                             self.d,
                             self.a,
                             self.x,
                             self.y,
                             self.z)[0:3, 3]
        elif arg == "orientation":
            return self.DK_n(self.theta,
                             self.alpha,
                             self.d,
                             self.a,
                             self.x,
                             self.y,
                             self.z)[0:3, 0:3]
        return None

    # fonction de calcul de la jacobienne symbolique et numérique lambdifiée
    def Jac_rob(self):

        Jac_v = self.DK_s[0:3, 3].jacobian(self.theta_sym)
        Jac_w = sp.Matrix.hstack(*[e[0:3, 2] for e in self.links_cumul])
        Jac = sp.Matrix.vstack(Jac_v, Jac_w)
        f = sp.lambdify((self.theta_sym,
                         self.alpha_sym,
                         self.d_sym,
                         self.a_sym, x, y, z), Jac, 'numpy')

        return Jac, f

    # résolution numérique de la Jacobienne
    def Jac_num(self):
        return self.Jac_n(self.theta,
                          self.alpha,
                          self.d,
                          self.a,
                          self.x,
                          self.y,
                          self.z)

    # résolution de la cinématique inverse (en cours d'écriture)
    def IK_rob(self, xt, yt, zt, eulert, lam):
        zi = []
        oi = []
        expr = []

        # vérification de l'existence d'une intersection commune aux trois derniers joints
        for i in range(3, 6):
            zi.append(self.links[i][0:3, 2])
            oi.append(self.links_cumul[i][0:3, 3])
            expr.append(oi + delta[i-3]*zi)

        for j in range(0,3):
            expr[j] = expr[j].subs({self.theta_sym: self.theta,
                                    self.alpha_sym: self.alpha,
                                    self.a_sym: self.a,
                                    self.d_sym: self.d,
                                    x: self.x,
                                    y: self.y,
                                    z: self.z})

        solution = sp.solve((expr[0], expr[1], expr[2]), delta)

        # si une solution existe, on résout analytiquement car présence d'un poignet sphérique
        if solution:
            R_pos = self.DK_s[:3, :3]
            P_pos = self.DK_s[:3, 3]

            R_fin_z = sp.Matrix([[sp.cos(euler[0]), -sp.sin(euler[0]), 0.0, 0.0],
                                 [sp.sin(euler[0]), sp.cos(euler[0]), 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])

            R_fin_y = sp.Matrix([[sp.cos(euler[1]), 0.0, sp.sin(euler[1]), 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [-sp.sin(euler[1]), 0.0, sp.cos(euler[1]), 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])

            R_fin_x = sp.Matrix([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, sp.cos(euler[2]), -sp.sin(euler[2]), 0.0],
                                 [0.0, sp.sin(euler[2]), sp.cos(euler[2]), 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])

            R_fin = R_fin_z * R_fin_y * R_fin_x #orientation finale en fonction de 3 angles d'euler

            # orientation de la position actuelle et désirée en quatternions
            quat_init = sp.Quaternion.from_rotation_matrix(R_pos)
            quat_fin = sp.Quaternion.from_rotation_matrix(R_fin)

            # équations position/orientation
            eq = [P_pos[0] - x_t,
                  P_pos[1] - y_t,
                  P_pos[2] - z_t,
                  quat_fin[0] - quat_init[0],
                  quat_fin[1] - quat_init[1],
                  quat_fin[2] - quat_init[2],
                  quat_fin[3] - quat_init[3]]

            # dictionnaire des éléments à substituer dans l'expression symbolique
            subs_dict = dict(zip(self.theta_sym, self.theta))
            subs_dict.update(zip(self.alpha_sym, self.alpha))
            subs_dict.update(zip(self.a_sym, self.a))
            subs_dict.update(zip(self.d_sym, self.d))
            subs_dict.update({x: self.x, y: self.y, z: self.z, x_t: xt, y_t: yt, z_t: zt})
            subs_dict.update(zip(euler, eulert))

            # liste des équations après substitution
            eq = [e.subs(subs_dict) for e in eq]

            # Résolution analytique des deux équations disjointes pos/rot
            IK_pos = sp.solve(eq[0:3], self.theta_sym[0:3])
            IK_rot = sp.solve(eq[3:6], self.theta_sym[3:6])

            return IK_pos, IK_rot # résol analytique: renvoie les theta pos et theta rot
        else:
            # on convertit les angles finaux en radians. RAJOUTER LA POSSIBILITE DE CHOISIR RADIANS OU DEGRES EN ARGUMENT ET CONVERSION ADAPTEE
            eulert_compute = np.radians(eulert)

            mat_dest = self.IK_n(eulert_compute, xt, yt, zt)

            curr_val = self.DK_num("pose")
            # calcul de l'erreur relative par transposée de la position actuelle * pose finale souhaitée
            Err = np.dot(np.transpose(curr_val), mat_dest)
            R = Err[:3, :3]
            T = Err[:3, 3]

            theta_to_log = np.arccos((np.trace(R) - 1.0) / 2)

            if theta_to_log < 1e-6:
                loga_rot = np.zeros((3, 3))
                axis_rot = np.zeros(3)
            else:
                loga_rot = (theta_to_log / (2 * np.sin(theta_to_log))) * (R - np.transpose(R))  # omega hat
                axis_rot = np.array([loga_rot[2, 1], loga_rot[0, 2], loga_rot[1, 0]])  # omega

            loga_pos = np.eye(3) - (0.5 * loga_rot) + (
                        1 - (theta_to_log * np.sin(theta_to_log) / (2 * (1 - np.cos(theta_to_log)))) * (
                            np.dot(loga_rot, loga_rot) / (theta_to_log ** 2)))

            twist = np.concatenate((np.dot(loga_pos, T), axis_rot))
            lambda_squared = lam
            J = self.Jac_num()
            m, n = np.shape(J)
            I = np.eye(m)
            ps_jac = np.dot(np.transpose(J), np.linalg.inv(np.dot(J, np.transpose(J)) + lambda_squared * I))

            self.theta = self.theta + 0.1 * (np.dot(ps_jac, twist))
        return None

    def cin_num_comp(self):
        x_plt = []
        y_plt = []
        z_plt = []
        axe_lab = ["seg1", "seg2", "seg3", "seg4", "seg5", "seg6", ]
        for i in range(0, self.nb_j):
            if i == 0:
                x_plt.append([0, self.DK_npart[0](self.theta[i], self.alpha[i],
                                                  self.d[i], self.a[i])[0, 3]])
                y_plt.append([0, self.DK_npart[0](self.theta[i], self.alpha[i],
                                                  self.d[i], self.a[i])[1, 3]])
                z_plt.append([0, self.DK_npart[0](self.theta[i], self.alpha[i],
                                                  self.d[i], self.a[i])[2, 3]])

            elif i == 1:
                x_plt.append([self.DK_npart[0](self.theta[0],
                                               self.alpha[0],
                                               self.d[0],
                                               self.a[0])[0, 3],
                              self.DK_npart[i](self.theta[0:i + 1],
                                               self.alpha[0:i + 1],
                                               self.d[0:i + 1],
                                               self.a[0:i + 1])[0, 3]])
                y_plt.append([self.DK_npart[i - 1](self.theta[0],
                                                   self.alpha[0],
                                                   self.d[0],
                                                   self.a[0])[1, 3],
                              self.DK_npart[i](self.theta[0:i + 1],
                                               self.alpha[0:i + 1],
                                               self.d[0:i + 1],
                                               self.a[0:i + 1])[1, 3]])
                z_plt.append([self.DK_npart[i - 1](self.theta[0],
                                                   self.alpha[0],
                                                   self.d[0],
                                                   self.a[0])[2, 3],
                              self.DK_npart[i](self.theta[0:i + 1],
                                               self.alpha[0:i + 1],
                                               self.d[0:i + 1],
                                               self.a[0:i + 1])[2, 3]])

            elif i > 1 and i < self.nb_j - 1:
                x_plt.append([self.DK_npart[i - 1](self.theta[0:i],
                                                   self.alpha[0:i],
                                                   self.d[0:i],
                                                   self.a[0:i])[0, 3],
                              self.DK_npart[i](self.theta[0:i + 1],
                                               self.alpha[0:i + 1],
                                               self.d[0:i + 1],
                                               self.a[0:i + 1])[0, 3]])
                y_plt.append([self.DK_npart[i - 1](self.theta[0:i],
                                                   self.alpha[0:i],
                                                   self.d[0:i],
                                                   self.a[0:i])[1, 3],
                              self.DK_npart[i](self.theta[0:i + 1],
                                               self.alpha[0:i + 1],
                                               self.d[0:i + 1],
                                               self.a[0:i + 1])[1, 3]])
                z_plt.append([self.DK_npart[i - 1](self.theta[0:i],
                                                   self.alpha[0:i],
                                                   self.d[0:i],
                                                   self.a[0:i])[2, 3],
                              self.DK_npart[i](self.theta[0:i + 1],
                                               self.alpha[0:i + 1],
                                               self.d[0:i + 1],
                                               self.a[0:i + 1])[2, 3]])
            else:
                x_plt.append([self.DK_npart[i - 1](self.theta[0:i],
                                                   self.alpha[0:i],
                                                   self.d[0:i],
                                                   self.a[0:i])[0, 3],
                              self.DK_npart[i](self.theta[0:i + 1],
                                               self.alpha[0:i + 1],
                                               self.d[0:i + 1],
                                               self.a[0:i + 1],
                                               self.x,
                                               self.y,
                                               self.z)[0, 3]])
                y_plt.append([self.DK_npart[i - 1](self.theta[0:i],
                                                   self.alpha[0:i],
                                                   self.d[0:i],
                                                   self.a[0:i])[1, 3],
                              self.DK_npart[i](self.theta[0:i + 1],
                                               self.alpha[0:i + 1],
                                               self.d[0:i + 1],
                                               self.a[0:i + 1],
                                               self.x,
                                               self.y,
                                               self.z)[1, 3]])
                z_plt.append([self.DK_npart[i - 1](self.theta[0:i],
                                                   self.alpha[0:i],
                                                   self.d[0:i],
                                                   self.a[0:i])[2, 3],
                              self.DK_npart[i](self.theta[0:i + 1],
                                               self.alpha[0:i + 1],
                                               self.d[0:i + 1],
                                               self.a[0:i + 1],
                                               self.x,
                                               self.y,
                                               self.z)[2, 3]])
        R = self.DK_num("orientation")
        arg1 = x_plt[self.nb_j - 1][1], y_plt[self.nb_j - 1][1], z_plt[self.nb_j - 1][1], *R[:, 0]
        arg2 = x_plt[self.nb_j - 1][1], y_plt[self.nb_j - 1][1], z_plt[self.nb_j - 1][1], *R[:, 1]
        arg3 = x_plt[self.nb_j - 1][1], y_plt[self.nb_j - 1][1], z_plt[self.nb_j - 1][1], *R[:, 2]
        return x_plt, y_plt, z_plt, R, arg1, arg2, arg3