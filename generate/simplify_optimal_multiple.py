from casadi import MX, DM, vertcat, horzcat, veccat, norm_2, dot, mtimes, nlpsol, diag, repmat, sum1
import casadi as ca
import numpy as np
from pygments import lex
import matplotlib.pyplot as plt



class Planner:
    def __init__(self,wp):
        ## changeable
        self.N = 30
        self.NX = 1
        self.a_max = 5
        self.wp = wp
        self.vel_guess = 5.0
        self.tol = 0.1

        self.NW = len(self.wp)
        self.dis = abs(self.wp[0])
        for i in range(len(self.wp)-1):
            self.dis += abs(self.wp[i+1]-self.wp[i])

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
        v_init = MX.sym('v_init', self.NX)
        x += [v_init]
        xg += [0]
        g += [v_init]
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
        self.vx = v_init
        self.tx = t_init
        self.lamx = lam_init
        self.mux = mu_init
        for i in range(self.N):
            lx = self.sx
            lv = self.vx
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
            self.vx = MX.sym('vx' + str(i), self.NX)
            x += [self.sx]
            xg += [0]
            x += [self.vx]
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
                diff = self.sx - self.wp[j]
                g += [self.mux[j] * (diff**2-tau[j])]
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
            g += [(self.sx - lx)-(lv + self.vx) * dt / 2.0]
            lb += [0]
            ub += [0]
            ##v&a
            g += [(-lv + self.vx) / dt]
            lb += [-self.a_max]
            ub += [self.a_max]


        t_end = MX.sym('t_end', self.NX)
        x += [t_end]
        xg += [0]
        x_end = MX.sym('x_end', self.NX)
        x += [x_end]
        xg += [self.wp[-1]]
        g += [x_end]
        lb += [self.wp[-1]]
        ub += [self.wp[-1]]
        v_end = MX.sym('v_end', self.NX)
        x += [v_end]
        xg += [0]
        g += [v_end]
        lb += [0]
        ub += [0]

        lam_end = MX.sym('lam_end', self.NW)
        x += [lam_end]
        xg += [0]*self.NW
        g += [lam_end]
        lb += [0]*self.NW
        ub += [0]*self.NW

        
        # mu_end = MX.sym('mu_end', self.NW)
        # x += [mu_end]*self.NW
        # xg += [0]*self.NW

        # tau_end = MX.sym('tau_end', self.NW)
        # x += [tau_end]*self.NW
        # xg += [0]*self.NW

        lt = self.tx
        lx = self.sx
        lv = self.vx
        llam = self.lamx
        lmu = self.mux
        self.tx = t_end
        self.sx = x_end
        self.vx = v_end
        self.lamx = lam_end
        g += [self.tx - lt - dt]
        lb += [0]
        ub += [0] 
        ##x&v
        g += [(self.sx - lx)-(lv + self.vx) * dt / 2.0]
        lb += [0]
        ub += [0]
        ##v&a
        g += [(-lv + self.vx) / dt]
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
        return self.x_sol, dt, self.N, self.NW
    

if __name__ == "__main__":
    # plan = Planner()
    # # print(plan.solve())
    # x_sol, dt, N, NW = plan.solve()
    # print(x_sol[0])
    # t_x = ca.DM(np.linspace(0, x_sol[0], N+2, True))
    # s_x = []
    # for i in range(N+2):
    #     s_x += [x_sol[2+i*3*(NW + 1)]]

    f = open("/home/zhoujin/trajectory-generation/trajectory/gate1.txt",'w')
    wp = [1,1,1,1]
    for x0i in range(2):
        for x1i in range(2):
            for y0i in range(2):
                for y1i in range(2):
                    wp[0] = 3 + x0i
                    wp[1] = 1 + x1i
                    wp[2] = 5 + y0i
                    wp[3] = 1 + y1i
            
                    plan = Planner(wp)
                    x_sol, dt, N, NW = plan.solve()
                    # print(x_sol[0])
                    t_x = ca.DM(np.linspace(0, x_sol[0], N+2, True))
                    s_t = []
                    s_x = []                   
                    for i in range(N+1):
                        s_t += [x_sol[1+i*3*(NW + 1)]]
                        s_x += [x_sol[2+i*3*(NW + 1)]]
                        f.write(str(wp[0])+','+str(wp[1])+','+str(wp[2])+','+str(wp[3])+',')
                        f.write(str(x_sol[1+i*3*(NW + 1)])+',')
                        f.write(str(x_sol[2+i*3*(NW + 1)])+',')
                        f.write(str(x_sol[3+i*3*(NW + 1)])+',')
                        for j in range(NW):
                            f.write(str(x_sol[4+j+i*3*(NW + 1)])+',')
                        f.write(str(x_sol[2+(i+1)*3*(NW + 1)])+',')
                        f.write(str(x_sol[3+(i+1)*3*(NW + 1)])+',')
                        for j in range(NW-1):
                            f.write(str(x_sol[4+j+(i+1)*3*(NW + 1)])+',')
                        f.write(str(x_sol[4+(NW-1)+(i+1)*3*(NW + 1)])+'\n')
                    plt.plot(s_t, s_x)
    f.close()

    # print(s_x)
    # print(t_x)
    fig = plt.figure(1)
    plt.title("Position - Time     Multiple Waypoints")
    plt.xlabel("Time /s")
    plt.ylabel("Position /m")
    plt.show()


    