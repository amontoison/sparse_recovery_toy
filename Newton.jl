using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;
include("mapping.jl")
include("Mtgeneration.jl")
include("Mperptz.jl")
include("fgNT.jl")
include("backtracking.jl")

# Newton's method

# It helps to find the minimum of \phi at given t.
# In the interior point method, the Newton method start at the minimun of last t

# @param t The parameter of central path in interior point method (scalar)
# @param beta_init Starting beta (a 1-dimensional vector)
# @param c_init Starting c (a 1-dimensional vector)
# (beta_init, c_init) is feasible
# @param delta_beta Newton direction in beta (a 1-dimensional vector)
# @param delta_c Newton direction in c (a 1-dimensional vector)
# @param paramNT A tuple consist of 2 entries: (eps_NT, paramLS)
# eps_NT Tolerance in Newton step (scalar)
# paramLS A tuple consist of 3 entries: (alpha_LS, gamma_LS, paramf)
# alpha_LS The reduction fraction parameter (scalar, 0<alpha_LS<1)
# gamma_LS The update factor of t_NT parameter (scalar, 0<gamma_LS<1)
# paramf A tuple consist of 3 entries: (Mt, M_perptz, d)
# Mt M^{\top}: The transpose of M
# M_perptz M_{\perp}^{\top}*z
# d The constraint on the l1 norm of beta (scalar)
# @param timevec A vector which record the time cost of every Newton direction computation
# The imput should be the vector from the last Newton step


# @return beta, c The minimum of \phi at given t (two vectors)
# @return timevec A vector which record the time cost of
# every Newton direction computation for all the Newton step till now

function NT_mtd(t, beta_init, c_init, paramNT, timevec)
    eps_NT = paramNT[1];
    paramLS = paramNT[2];
    paramf = paramLS[3];

    beta = beta_init;
    c = c_init;
    timetemp = 0;

    count = 0;

    while(true)
        count = count + 1;
        if(count>1000)
            println("Newton's method doesn't terminate")
        end
        gradb, gradc = fgrad(t, beta, c, paramf);

        # compute the Newton direction and the time cost
        timetemp = @elapsed begin
        delta_beta, delta_c = NT_direction(t, beta, c, gradb, gradc, paramf);
        end

        # update the time record vector
        timevec = [timevec; timetemp];

        # compute increment
        lambda2 = -sum((gradb'*delta_beta).+(gradc'*delta_c));

        # check terminate condition
        if ((lambda2/2) <= eps_NT)
            break;
        end

        # use bracktracking line search to compute step size
        t_NT = backtracking(t, beta, c, delta_beta, delta_c, paramLS);

        # update beta, c
        beta = beta.+(t_NT.*delta_beta);
        c = c.+(t_NT.*delta_c);
    end

    return beta, c, timevec
end
