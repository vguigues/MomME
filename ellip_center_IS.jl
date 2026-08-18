
include("gradient_ImgSmooth.jl")

function ellip_center_IS(xk::AbstractMatrix,y::AbstractMatrix,lambda::Real,mxitr::Int,tol::Real,omega::Real)
    
    gk = gradient_ImgSmooth(xk, y, lambda)
    start_time = time()
    k = 0
    tol_ME=1e-7
    nrmG = norm(gk)
    xtemp=zeros(size(xk))
    gtemp=zeros(size(xk)) 

    while nrmG > tol
        wk = gradient_ImgSmooth(gk, y, lambda) + y   
        gtw = sum(gk.*wk)
        ng2 = nrmG^2
        tk = 2*(ng2/gtw)
        yk = xk - tk*gk
        rk = gk - tk*wk
        nrmR = norm(rk)

        if nrmR > tol
           if abs(nrmR-nrmG)<tol_ME*max(1,abs(nrmR)) 
                xtemp = (xk + yk) / 2
                gtemp = 0.5*(gk + rk)
                nrmG=norm(gtemp)
                xk=xtemp
                break
           else 
                Awk  = gradient_ImgSmooth(wk, y, lambda) + y        
                Ark = wk - tk*Awk      
                rtAr = sum(rk.*Ark)     
                gtAr = sum(gk.*Ark)

                delta  = rtAr*gtw - gtAr^2 
        
                rtg = sum(rk.*gk)   
                alpha = (rtg*gtAr - ng2*rtAr)/delta
                beta  = (-rtg*gtw + ng2*gtAr)/delta
				xtemp=xk+alpha*gk + beta*rk
				gtemp=A*xtemp-b
            end 
            if k==0
			   xk=xtemp
               gk=gtemp
			else
               xk = (1-omega)*xk + omega*xtemp
			   gk = (1-omega)*gk + omega*gtemp
			end	
        else
            xk = yk    
            nrmG = nrmR
            k=k+1
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

