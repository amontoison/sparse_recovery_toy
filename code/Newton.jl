include("fgNT.jl")
include("backtracking.jl")

function NT_mtd(t, beta_init, c_init, eps_NT, paramLS, paramf)
    beta = beta_init;
    c = c_init;

    count = 0;

    while(true)
        count = count + 1;
        if(count>1000)
            println("Newton's method doesn't terminate, t = ", t)
        end

        gradb, gradc = fgrad2(t, beta, c, paramf);
        delta_beta, delta_c = CG(t, beta, c, gradb, gradc, paramf);
        #delta_beta1, delta_c1 = NTdir(t, beta, c, gradb, gradc, paramf);
        #println("normdeltab:", norm(delta_beta.-delta_beta1))
        #println("normdeltac:", norm(delta_c.-delta_c1))

        # compute increment
        lambda2 = -sum((gradb'*delta_beta).+(gradc'*delta_c));

        # check terminate condition
        if ((lambda2/2) <= eps_NT)
            break;
        end

        # use bracktracking line search to compute step size
        t_NT = backtracking(t, beta, c, delta_beta, delta_c, paramLS, paramf);

        # update beta, c
        beta = beta.+(t_NT.*delta_beta);
        c = c.+(t_NT.*delta_c);
    end

    return beta, c
end
