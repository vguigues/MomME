function MEM_IS(xk::AbstractMatrix,y::AbstractMatrix,lambda::Real,mxitr::Int,tol::Real)
    
    gk = gradient_ImgSmooth(xk, y, lambda)
    start_time = time()
    k = 0
    nrmG = norm(gk)
    xold = xk  
    gold = gk
    while nrmG > tol
        xk_prev = xk           
        gk_prev = gk
        wk = gradient_ImgSmooth(gk, y, lambda) + y
        gtw = sum(gk.*wk)  
        ng2 = nrmG^2
        tk = 2*(ng2/gtw)
        yk = xk - tk*gk
        rk = gk - tk*wk
        nrmR = norm(rk)

        if nrmR > tol
            Awk  = gradient_ImgSmooth(wk, y, lambda) + y        
            Ark = wk - tk*Awk      
            rtAr = sum(rk.*Ark)     
            gtAr = sum(gk.*Ark)

            delta  = rtAr*gtw - gtAr^2 
            if abs(delta) > 0 
                rtg = sum(rk.*gk)   

                alpha = (rtg*gtAr - ng2*rtAr)/delta
                beta  = (-rtg*gtw + ng2*gtAr)/delta

                xk_EM = xk + alpha*gk + beta*rk
                gk_EM = gk + alpha*wk + beta*Ark

            else
                xk_EM = (xk + yk) / 2
                gk_EM = 0.5*(gk + rk)
            end
        else
            xk = yk    
            nrmG = nrmR
            break
        end

        sk = xold - xk_EM                  
        gsk = gold - gk_EM 
        mu = -sum(sk.*gk_EM)/sum(sk.*gsk)
        xk = xk_EM + mu*sk
        gk = gk_EM + mu*gsk
        

        xold = xk_prev  
        gold = gk_prev        
        nrmG = norm(gk)
        k = k + 1

        if k > mxitr
            break
        end
    end
    elapsed = time() - start_time

    return xk, elapsed, nrmG, k-1
end

