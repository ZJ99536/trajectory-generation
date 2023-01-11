param N := 20;
param dis := 6.0;
param a_max := 5.0;

var x{j in 1..N};
var v{j in 1..N};
var t;

minimize time_optimal: t;

subject to t_limit:
0.1 <= t;

subject to pos_init:
x[1] = 0;
subject to vel_init:
v[1] = 0;
subject to pos_end:
x[N] = dis;
subject to vel_end:
v[N] = 0;

subject to limit_xv {i in 2..N}:
x[i] - x[i-1] = (v[i] + v[i-1]) * (t / (N-1)) / 2.0;

subject to limit_va {i in 2..N}:
-a_max <= (v[i] - v[i-1]) / (t / (N-1)) <= a_max;




