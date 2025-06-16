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

        # initialisation des symboles propres
        self.theta_sym = sp.symbols(f'theta1:{self.nb_j+1}')
        self.alpha_sym = sp.symbols(f'alpha1:{self.nb_j+1}')
        self.a_sym = sp.symbols(f'a1:{self.nb_j+1}')
        self.d_sym = sp.symbols(f'd1:{self.nb_j+1}')

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
    def set_theta(self, t, i=0):
        if i == 0:
            self.theta = t
        else:
            self.theta[i-1] = t

    def set_alpha(self, a, i=0):
        if i == 0:
            self.alpha = a
        else:
            self.alpha[i-1] = a

    def set_a(self, a, i=0):
        if i == 0:
            self.a = a
        else:
            self.a[i-1] = a

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

    # résolution de la cinématique inverse
    def IK_rob(self, xt, yt, zt, eulert):
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
        return None

    # segments computing and adding to the plot
    def cin_num_comp(self, axe):
        x_plt = []
        y_plt = []
        z_plt = []
        axe_lab = ["seg1", "seg2", "seg3", "seg4", "seg5", "seg6",]
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
                              self.DK_npart[i](self.theta[0:i+1],
                                               self.alpha[0:i+1],
                                               self.d[0:i+1],
                                               self.a[0:i+1])[0, 3]])
                y_plt.append([self.DK_npart[i-1](self.theta[0],
                                                 self.alpha[0],
                                                 self.d[0],
                                                 self.a[0])[1, 3],
                              self.DK_npart[i](self.theta[0:i+1],
                                               self.alpha[0:i+1],
                                               self.d[0:i+1],
                                               self.a[0:i+1])[1, 3]])
                z_plt.append([self.DK_npart[i-1](self.theta[0],
                                                 self.alpha[0],
                                                 self.d[0],
                                                 self.a[0])[2, 3],
                              self.DK_npart[i](self.theta[0:i+1],
                                               self.alpha[0:i+1],
                                               self.d[0:i+1],
                                               self.a[0:i+1])[2, 3]])

            elif i > 1 and i < self.nb_j-1:
                x_plt.append([self.DK_npart[i-1](self.theta[0:i],
                                                 self.alpha[0:i],
                                                 self.d[0:i],
                                                 self.a[0:i])[0, 3],
                              self.DK_npart[i](self.theta[0:i+1],
                                               self.alpha[0:i+1],
                                               self.d[0:i+1],
                                               self.a[0:i+1])[0, 3]])
                y_plt.append([self.DK_npart[i-1](self.theta[0:i],
                                                 self.alpha[0:i],
                                                 self.d[0:i],
                                                 self.a[0:i])[1, 3],
                              self.DK_npart[i](self.theta[0:i+1],
                                               self.alpha[0:i+1],
                                               self.d[0:i+1],
                                               self.a[0:i+1])[1, 3]])
                z_plt.append([self.DK_npart[i-1](self.theta[0:i],
                                                 self.alpha[0:i],
                                                 self.d[0:i],
                                                 self.a[0:i])[2, 3],
                              self.DK_npart[i](self.theta[0:i+1],
                                               self.alpha[0:i+1],
                                               self.d[0:i+1],
                                               self.a[0:i+1])[2, 3]])
            else:
                x_plt.append([self.DK_npart[i-1](self.theta[0:i],
                                                 self.alpha[0:i],
                                                 self.d[0:i],
                                                 self.a[0:i])[0, 3],
                              self.DK_npart[i](self.theta[0:i+1],
                                               self.alpha[0:i+1],
                                               self.d[0:i+1],
                                               self.a[0:i+1],
                                               self.x,
                                               self.y,
                                               self.z)[0, 3]])
                y_plt.append([self.DK_npart[i-1](self.theta[0:i],
                                                 self.alpha[0:i],
                                                 self.d[0:i],
                                                 self.a[0:i])[1, 3],
                              self.DK_npart[i](self.theta[0:i+1],
                                               self.alpha[0:i+1],
                                               self.d[0:i+1],
                                               self.a[0:i+1],
                                               self.x,
                                               self.y,
                                               self.z)[1, 3]])
                z_plt.append([self.DK_npart[i-1](self.theta[0:i],
                                                 self.alpha[0:i],
                                                 self.d[0:i],
                                                 self.a[0:i])[2, 3],
                              self.DK_npart[i](self.theta[0:i+1],
                                               self.alpha[0:i+1],
                                               self.d[0:i+1],
                                               self.a[0:i+1],
                                               self.x,
                                               self.y,
                                               self.z)[2, 3]])
            axe.plot(x_plt[i], y_plt[i], z_plt[i], label=axe_lab[i])
        R = self.DK_num("orientation")
        axe.quiver(x_plt[self.nb_j-1][1], y_plt[self.nb_j-1][1],
                   z_plt[self.nb_j-1][1],
                   *R[:, 0],
                   length=5,
                   color='r')
        axe.quiver(x_plt[self.nb_j-1][1], y_plt[self.nb_j-1][1],
                   z_plt[self.nb_j-1][1],
                   *R[:, 1],
                   length=5,
                   color='g')
        axe.quiver(x_plt[self.nb_j-1][1], y_plt[self.nb_j-1][1],
                   z_plt[self.nb_j-1][1],
                   *R[:, 2],
                   length=5,
                   color='b')
    
    # plot de la position du robot et de la position des joints
    def plot(self):

        # labels axes

        # définition de l'espace de plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        fig.subplots_adjust(bottom=0.4)

        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(0, 50)

        # remplissages du ax avec les segments du robots
        self.cin_num_comp(ax)
        sliders = []
        for i in range(self.nb_j):
            axthet = fig.add_axes([0.25, 0.35 - i*0.03, 0.65, 0.02])
            slider = wd.Slider(
                axthet,
                f'theta[{i}]',
                -np.pi,
                np.pi,
                valinit=0.0
            )
            sliders.append(slider)

        def update(val):
            ax.cla()
            theta_vals = [s.val for s in sliders]
            self.set_theta(theta_vals)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_zlim(0, 50)
            self.cin_num_comp(ax)
            fig.canvas.draw_idle()

        for s in sliders:
            s.on_changed(update)

        ax.legend()
        plt.show()
