from cmath import cos
from casadi import MX, DM, vertcat, horzcat, veccat, norm_2, dot, mtimes, nlpsol, diag, repmat, sum1, sin
import casadi as ca
from cv2 import sqrt
import numpy as np
from pygments import lex
import matplotlib.pyplot as plt



class Planner:
    def __init__(self,wpx,wpy):
        ## changeable
        self.N = 30
        self.NX = 1
        self.a_max = 5
        self.wpx = [2.0, 6.0]
        self.wpy = [5.0, 2.0]
        self.wpx = wpx
        self.wpy = wpy
        self.vel_guess = 5.0
        self.tol = 0.3

        self.NW = len(self.wpx)
        self.dis = 15
        self.dis = ((self.wpx[0])**2 + (self.wpy[0])**2)**0.5
        for i in range(len(self.wpx)-1):
            self.dis += ((self.wpx[i+1]-self.wpx[i])**2 + (self.wpy[i+1]-self.wpy[i])**2)**0.5

        # Problem variables
        self.x = []
        self.xg = []
        self.g = []
        self.lb = []
        self.ub = []
        self.J = []

    def solve(self):

        x = []
        xg = []
        g = []
        lb = []
        ub = []
        J = 0  
          
        t = MX.sym('t', 1)
        x += [t]
        xg += [self.dis/self.vel_guess]
        g += [t]
        lb += [0.1]
        ub += [150]
        J = t

        t_init = MX.sym('t_init', self.NX)
        x += [t_init]
        xg += [0]
        g += [t_init]
        lb += [0]
        ub += [0]
        x_init = MX.sym('x_init', self.NX)
        x += [x_init]
        xg += [0]
        g += [x_init]
        lb += [0]
        ub += [0]
        y_init = MX.sym('y_init', self.NX)
        x += [y_init]
        xg += [0]
        g += [y_init]
        lb += [0]
        ub += [0]
        vx_init = MX.sym('vx_init', self.NX)
        x += [vx_init]
        xg += [0]
        g += [vx_init]
        lb += [0]
        ub += [0]
        vy_init = MX.sym('vy_init', self.NX)
        x += [vy_init]
        xg += [0]
        g += [vy_init]
        lb += [0]
        ub += [0]

        lam_init = MX.sym('lam_init', self.NW)
        x += [lam_init]
        xg += [1]*self.NW
        g += [lam_init]
        lb += [1]*self.NW
        ub += [1]*self.NW
        mu_init = MX.sym('mu_init', self.NW)
        x += [mu_init]
        xg += [0]*self.NW
        g += [mu_init]
        lb += [0]*self.NW
        ub += [0]*self.NW
        tau = MX.sym('tau', self.NW)
        x += [tau]
        xg += [0] * self.NW
        g += [tau]
        lb += [0]*self.NW
        ub += [self.tol**2]*(self.NW)

        dt = t / (self.N + 1)

        self.sx = x_init
        self.vx = vx_init
        self.sy = y_init
        self.vy = vy_init
        self.tx = t_init
        self.lamx = lam_init
        self.mux = mu_init
        for i in range(self.N):
            lx = self.sx
            lvx = self.vx
            ly = self.sy
            lvy = self.vy
            lt = self.tx
            llam = self.lamx
            lmu = self.mux
            self.tx = MX.sym('t'+str(i),1)
            x += [self.tx]
            xg += [0]
            g += [self.tx - lt - dt]
            lb += [0]
            ub += [0]            
            self.sx = MX.sym('sx' + str(i), self.NX)
            self.sy = MX.sym('sy' + str(i), self.NX)
            self.vx = MX.sym('vx' + str(i), self.NX)
            self.vy = MX.sym('vy' + str(i), self.NX)
            x += [self.sx]
            xg += [0]
            x += [self.sy]
            xg += [0]
            x += [self.vx]
            xg += [0]
            x += [self.vy]
            xg += [0]
            self.lamx = MX.sym('lam' + str(i), self.NW)
            self.mux = MX.sym('mu' + str(i), self.NW)
            x += [self.lamx]
            xg += [1]*self.NW
            g += [self.lamx]
            lb += [0]*self.NW
            ub += [1]*self.NW
            x += [self.mux]
            xg += [0]*self.NW
            g += [self.mux]
            lb += [0]*self.NW
            ub += [1]*self.NW
            tau = MX.sym('tau'+str(i), self.NW)
            x += [tau]
            xg += [0] * self.NW
            g += [tau]
            lb += [0]*self.NW
            ub += [self.tol**2]*(self.NW)

            for j in range(self.NW):
                diff = (self.sx - self.wpx[j])**2 + (self.sy - self.wpy[j])**2
                # if j == 0:
                #     diff = (self.sx - self.wpx[j])**2 + (self.sy - self.wpy[j]*(self.tx-1))**2
                # else:
                #     diff = (self.sx - self.wpx[j]*sin(self.tx))**2 + (self.sy - self.wpy[j])**2
                g += [self.mux[j] * (diff-tau[j])]
                lb += [0]
                ub += [0.01]

                if j < self.NW-1:
                    g += [self.lamx[j+1]-self.lamx[j]]
                    lb += [0]
                    ub += [1]
            
            g += [self.lamx - llam + lmu]
            lb += [0]*self.NW
            ub += [0]*self.NW

            ##x&v
            g += [(self.sx - lx)-(lvx + self.vx) * dt / 2.0]
            lb += [0]
            ub += [0]
            g += [(self.sy - ly)-(lvy + self.vy) * dt / 2.0]
            lb += [0]
            ub += [0]
            ##v&a
            g += [(-lvx + self.vx) / dt]
            lb += [-self.a_max]
            ub += [self.a_max]
            g += [(-lvy + self.vy) / dt]
            lb += [-self.a_max]
            ub += [self.a_max]


        t_end = MX.sym('t_end', self.NX)
        x += [t_end]
        xg += [0]
        x_end = MX.sym('x_end', self.NX)
        x += [x_end]
        xg += [self.wpx[-1]]
        # g += [x_end-self.wpx[-1]*sin(t_end)]
        g += [x_end]
        lb += [self.wpx[-1]]
        ub += [self.wpx[-1]]
        y_end = MX.sym('y_end', self.NX)
        x += [y_end]
        xg += [self.wpy[-1]]
        g += [y_end]
        lb += [self.wpy[-1]]
        ub += [self.wpy[-1]]
        vx_end = MX.sym('vx_end', self.NX)
        x += [vx_end]
        xg += [0]
        vy_end = MX.sym('vy_end', self.NX)
        x += [vy_end]
        xg += [0]


        lam_end = MX.sym('lam_end', self.NW)
        x += [lam_end]
        xg += [0]*self.NW
        g += [lam_end]
        lb += [0]*self.NW
        ub += [0]*self.NW

        lx = self.sx
        lvx = self.vx
        ly = self.sy
        lvy = self.vy
        lt = self.tx
        llam = self.lamx
        lmu = self.mux
        self.tx = t_end
        self.sx = x_end
        self.vx = vx_end
        self.sy = y_end
        self.vy = vy_end
        self.lamx = lam_end
        g += [self.tx - lt - dt]
        lb += [0]
        ub += [0] 
        ##x&v
        g += [(self.sx - lx)-(lvx + self.vx) * dt / 2.0]
        lb += [0]
        ub += [0]
        g += [(self.sy - ly)-(lvy + self.vy) * dt / 2.0]
        lb += [0]
        ub += [0]
        ##v&a
        g += [(-lvx + self.vx) / dt]
        lb += [-self.a_max]
        ub += [self.a_max]
        g += [(-lvy + self.vy) / dt]
        lb += [-self.a_max]
        ub += [self.a_max]
        ##lam&mu
        g += [self.lamx - llam + lmu]
        lb += [0]*self.NW
        ub += [0]*self.NW

        # Reformat
        self.x = vertcat(*x)
        if not self.xg:
            self.xg = xg
        self.xg = veccat(*self.xg)
        self.g = vertcat(*g)
        self.lb = veccat(*lb)
        self.ub = veccat(*ub)
        self.J = J

        # Construct Non-Linear Program
        self.nlp = {'f': self.J, 'x': self.x, 'g': self.g}

        self.solver = nlpsol('solver', 'ipopt', self.nlp)
        self.solution = self.solver(x0=self.xg, lbg=self.lb, ubg=self.ub)
        self.x_sol = self.solution['x'].full().flatten()
        return self.x_sol, dt, self.N, self.NW, self.wpx, self.wpy
    

if __name__ == "__main__":
    fig = plt.figure(1)
    plt.title("Two Gates")
    plt.xlabel("Position x[m]")
    plt.ylabel("Position y[m]")
    f = open("/home/zhoujin/trajectory-generation/trajectory/gatedd.txt",'w')
    wpx = [1,1]
    wpy = [1,1]
    for x0i in range(-2,2):
        for x1i in range(-2,2):
            for y0i in range(-2,2):
                for y1i in range(-2,2):
                    wpx[0] = 2.0 + x0i * 0.1
                    wpx[1] = 1.0 + x1i * 0.1
                    wpy[0] = 2.0 + y0i * 0.1
                    wpy[1] = 1.0 + y1i * 0.1
            
                    plan = Planner(wpx,wpy)
                    x_sol, dt, N, NW, wpx, wpy = plan.solve()
                    # print(x_sol[0])
                    t_x = ca.DM(np.linspace(0, x_sol[0], N+2, True))
                    s_x = []
                    s_y = []                   
                    for i in range(N+2):
                        s_x += [x_sol[2+i*(3*NW + 5)]]
                        s_y += [x_sol[3+i*(3*NW + 5)]]
                        f.write(str(wpx[0]-x_sol[2+i*(3*NW + 5)])+','+str(wpy[0]-x_sol[2+i*(3*NW + 5)])+','+str(wpx[1]-x_sol[2+i*(3*NW + 5)])+','+str(wpy[1]-x_sol[2+i*(3*NW + 5)])+',')
                        f.write(str(x_sol[2+i*(3*NW + 5)])+',')
                        f.write(str(x_sol[3+i*(3*NW + 5)])+',')
                        f.write(str(x_sol[6+i*(3*NW + 5)])+',')
                        f.write(str(x_sol[7+i*(3*NW + 5)])+',')
                        f.write(str(x_sol[4+i*(3*NW + 5)])+',')
                        f.write(str(x_sol[5+i*(3*NW + 5)]))
                        
                        f.write('\n')
                    
                    plt.plot(s_x, s_y)
    f.close()
    plt.show()


    