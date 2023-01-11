set waypoints;
param wp {j in waypoints};
param NW;
param N;
param a_max;
param tol;

var x{j in 1..N};
var v{j in 1..N};
var lam{i in 1..NW, j in 1..N};
var mu{i in 1..NW, j in 1..N};
var tau{i in 1..NW};
var t;

minimize time_optimal: t;

subject to t_limit:
0.1 <= t;

subject to pos_init:
x[1] = 0;
subject to vel_init:
v[1] = 0;
subject to lam_init {i in 1..NW}:
lam[i,1] = 1;
subject to mu_init {i in 1..NW}:
mu[i,1] = 0;
subject to pos_end:
x[N] = wp[NW];
subject to vel_end:
v[N] = 0;
subject to lam_end {i in 1..NW}:
lam[i,N] = 0;

subject to limit_xv {i in 2..N}:
x[i] - x[i-1] = (v[i] + v[i-1]) * (t / (N-1)) / 2.0;
subject to limit_va {i in 2..N}:
-a_max <= (v[i] - v[i-1]) / (t / (N-1)) <= a_max;
subject to limit_lam_mu {i in 1..NW, j in 2..N}:
lam[i,j] = lam[i,j-1] - mu[i, j-1];

subject to limit_lam {i in 1..NW, j in 1..N}:
0 <= lam[i,j] <= 1;
subject to limit_mu {i in 1..NW, j in 1..N}:
0 <= mu[i,j] <= 1;
subject to limit_tau {i in 1..NW}:
0 <= tau[i] <= tol;

subject to order_lam {i in 2..NW, j in 1..N}:
0 <= lam[i,j] - lam[i-1,j] <= 1;
subject to reach_wps {i in 1..NW, j in 1..N}:
0 <= mu[i,j]*((x[j] - wp[i])*(x[j] - wp[i])-tau[i]) <= 0.01;




