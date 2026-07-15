function ellip_center_IS(xk::AbstractMatrix,y::AbstractMatrix,lambda::Real,mxitr::Int,tol::Real,omega::Real)
    
    gk = gradient_ImgSmooth(xk, y, lambda)
    start_time = time()
    k = 0
    nrmG = norm(gk)
    counter = 0
    while nrmG > tol
        wk = gradient_ImgSmooth(gk, y, lambda) + y   
        gtw = sum(sum(gk.*wk));  ng2 = nrmG^2
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

                xk = xk + (omega*alpha)*gk + (omega*beta)*rk
                gk = gk + (omega*alpha)*wk + (omega*beta)*Ark

            else
                counter = counter + 1
                xtemp = (xk + yk) / 2
                gtemp = 0.5*(gk + rk)
                xk = (1-omega)*xk + omega*xtemp  
                gk = (1-omega)*gk + omega*gtemp
            end
        else
            xk = yk;    nrmG = nrmR
            break
        end
        nrmG = norm(gk)
        k = k + 1
        
        if k > mxitr
            break
        end
    end
    
    elapsed = time() - start_time
      
    return xk, elapsed, nrmG, k

end

