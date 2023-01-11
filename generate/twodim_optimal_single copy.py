from casadi import MX, DM, vertcat, horzcat, veccat, norm_2, dot, mtimes, nlpsol, diag, repmat, sum1
import casadi as ca
import numpy as np
from pygments import lex
import matplotlib.pyplot as plt



class Planner:
    def __init__(self):
        self.NW = 3
        self.N = 20
        self.NX = 1
        self.a_max = 5

        self.dis = -6.0
        self.vel_guess = 3.0

        # Problem variables
        self.x = []
        self.xg = []
        self.g = []
        self.lb = []
        self.ub = []
        self.J = []

    def solve(self):
        t = MX.sym('t', 1)
        # a = MX.sym('a', 1)
        # J = (t-3)**2
        # x = vertcat(t,a)
        # g = t + a

        x = []
        xg = []
        g = []
        lb = []
        ub = []
        J = 0      


        
        x += [t]
        xg += [abs(self.dis)/self.vel_guess]
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

        dt = t / (self.N + 1)

        self.sx = x_init
        self.vx = vx_init
        self.sy = y_init
        self.vy = vy_init
        self.tx = t_init
        for i in range(self.N):
            lx = self.sx
            lvx = self.vx
            ly = self.sy
            lvy = self.vy
            lt = self.tx
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
        xg += [self.dis]
        g += [x_end]
        lb += [self.dis]
        ub += [self.dis]
        y_end = MX.sym('y_end', self.NX)
        x += [y_end]
        xg += [self.dis]
        g += [y_end]
        lb += [self.dis]
        ub += [self.dis]
        vx_end = MX.sym('vx_end', self.NX)
        x += [vx_end]
        xg += [0]
        g += [vx_end]
        lb += [0]
        ub += [0]
        vy_end = MX.sym('vy_end', self.NX)
        x += [vy_end]
        xg += [0]
        g += [vy_end]
        lb += [0]
        ub += [0]

        lx = self.sx
        lvx = self.vx
        ly = self.sy
        lvy = self.vy
        lt = self.tx
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
        return self.x_sol, dt, self.N
    

if __name__ == "__main__":
    plan = Planner()
    # print(plan.solve())
    x_sol, dt, N = plan.solve()
    t_x = ca.DM(np.linspace(0, x_sol[0], N+2, True))
    s_x = []
    for i in range(N+2):
        s_x += [x_sol[3+i*5]]

    # print(s_x)
    # print(t_x)
    print(x_sol)
    plt.title("Position - Time     Single Waypoint")
    plt.xlabel("Time /s")
    plt.ylabel("Position /m")
    fig = plt.figure(1)
    plt.plot(t_x, s_x)
    # plt.scatter(t_x[-1], s_x[-1])
    plt.show()
    # x_sol = np.array(x_sol)
    # t_axis = np.arange(0,x_sol[0]+dt,dt) 
    # # x_axis = np.zeros(N+2)

    