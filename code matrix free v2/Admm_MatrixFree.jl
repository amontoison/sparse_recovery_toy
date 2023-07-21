# solve \min_||z - M_{\perp}^t*beta||_2 + lambda*||beta||_1

# @param Mt M^{\top}: The transpose of M
# @param M_perptz M_{\perp}^{\top}*z
# @param lambda The penalty parameter in LASSO (scalar)
# @param rho The parameter in ADMM (scalar)
# @param maxk The max number of iters in CG (scalar)
# @param tol The tolerance (scalar)


# @return z0 LASSO problem solution
# @return ADMMerrrecord The norm of the difference between the previous and current step
# It can be deleted. It is saved just for the use of checking convergence of ADMM


function cgADMM(Mt, M_perptz, lambda, rho, maxt = 10000, tol = 1e-3)
    n = size(Mt)[1];
    x0 = zeros(n);
    z0 = zeros(n);
    y0 = zeros(n);
    t = 0;
    err = 1;

    while((t<maxt) & (err>tol))
        b = M_perptz.+(rho.*z0).-y0;
        # update x
        x1 = cg(Mt, rho, b);
        # update z
        z1 = softthreshold.(x1 + y0/rho, lambda/rho);
        # update y
        y1 = y0 + rho*(x1 - z1);
        # check the convergence
        err = max(norm(x1 - x0, 2), norm(y1 - y0, 2), norm(z1 - z0, 2));
        x0 = x1;
        z0 = z1;
        y0 = y1;
        t = t + 1;

    end

    return z0, t
end

# compute (M_{\perp}^{T} M_{\perp}+\rho I)^{-1}*b

# @param Mt M^{\top}: The transpose of M
# @param rho The parameter in ADMM (scalar)
# @param b The vector b = M_{\perp}^t*z.+(rho*z_k).-y_k
# @param maxk The max number of iters in CG (scalar)
# @param tol The tolerance (scalar)


# @return x_0 An approximation to (M_{\perp}^{T} M_{\perp}+\rho I)^{-1}*b
# @return k The number of iterations in CG (can be deleted, saved just for the use of investigating CG and ADMM performance)

function cg(Mt, rho, b, missing_indices, non_missing_values, maxk = 1000, tol = 1e-6)
    n = size(b)[1];
    non_missing_indices = setdiff(1:n, missing_indices)
    r0 = zeros(n);
    x0 = zeros(n);
    r0 = rho.*x0 - b             #\rho I*x0 - b
    w0 = beta_to_DFT(1, n, b)
    w0 = ifft(w0).*sqrt(n)
    w0 = w0[non_missing_indices]  #w0 is Mperp*x0
    
    
    x0[non_missing_indices] .= w0
    w0 = M_perp_tz(x0, 1, n)
    r0 = w0 + r0;                 #r_0 = (M_{\perp}^T M_{\perp} + \rho I)*x0 - b
    p0 = (-1).*r0; #p_0
    temp = copy(p0)
    p0 = rho*p0
    # Ap0 = ((1+rho).*p0).-(Mt*(Mt'*p0)); #A*p_0

    w0 = beta_to_DFT(1, n, b)
    w0 = ifft(w0).*sqrt(n)
    w0 = w0[non_missing_indices]
    temp[non_missing_indices] .= w0
    w0 = M_perp_tz(temp, 1, n)       #M_{\perp}^T M_{\perp} p_0 
    Ap0 = w0 + p0 
    k = 0;
    epn = 1;

    while((k<maxk) & (epn>tol))
        alpha = dot(r0, r0)/dot(p0, Ap0);
        x1 = x0 + alpha*p0;
        r1 = r0 + alpha*Ap0;
        beta = dot(r1, r1)/dot(r0, r0);
        p1 = -r1 + beta*p0;

        x0 = x1;
        r0 = r1;
        p0 = p1;
        Ap0 = ((1+rho).*p0).-(Mt*(Mt'*p0));
        epn = norm(r0, 2);
        k = k + 1;
    end
    return x0
end

# add soft threshold

# @param x The input value (scalar)
# @param thre The threshold (scalar, thre >= 0)

# @details: if x > thre, then y = x - thre
#           if -thre <= x <= thre, then y = 0
#           if x<= -thre, then y = x + thre

# @return y The value after threshold (scalar)

function softthreshold(x, thre)
    if(x > thre)
        y = x - thre
    elseif(x < -thre)
        y = x + thre
    else
        y = 0
    end

    return(y)
end
