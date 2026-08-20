
include("gradient_ImgSmooth.jl")



function MEM_IS(xk::AbstractMatrix,y::AbstractMatrix,lambda::Real,mxitr::Int,tol::Real)
    
    gk = gradient_ImgSmooth(xk, y, lambda)
    start_time = time()
    k = 0
    nrmG = norm(gk)
    tol_ME=1e-7
    xtemp=zeros(size(xk))
    gtemp=zeros(size(xk)) 
    xprev=zeros(size(xk))
    gprev=zeros(size(xk)) 


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
                k=k+1
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
				gtemp=gk+alpha*wk+beta*Ark
            end 

            if k==0
			   xprev.=xk 
			   gprev.=gk	
			   xk.=xtemp
               gk.=gtemp
			else
				xaux=copy(xk) 
				gaux=copy(gk)
				sk=xtemp-xprev
				gsk = gtemp-gprev 
				mu = (sum(sk.*gtemp))/(sum(sk.*gsk))
				xk = (1-mu)*xtemp+mu*xprev
				gk = (1-mu)*gtemp+mu*gprev
				xprev.=xaux
				gprev.=gaux
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

    return xk, elapsed, nrmG, k-1
end

