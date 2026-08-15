
include("util.jl")

function ellip_center_with_momentum(A::AbstractMatrix, xk::Vector, b::Vector, mxitr::Int, tol::Float64)



	gk = A*xk - b
	start_time = time()
	k = 0
	tol_ME=1e-7
	nrmG = norm(gk)
    nrmT=nrmG
    xtemp=zeros(size(xk))
    gtemp=zeros(size(xk)) 
    xprev=zeros(size(xk))
    gprev=zeros(size(xk)) 



	while nrmG > tol && nrmT>tol
		wk = A*gk
		gtw = gk'*wk
		ng2 = nrmG^2
		tk = 2*(ng2/gtw)
		yk = xk - tk*gk
		rk = gk - tk*wk
		nrmR = norm(rk)

		if nrmR > tol
			# if abs(nrmR-nrmG)<tol_ME
		    if abs(nrmR-nrmG)<tol_ME*max(1,abs(nrmR))
				println("Entering the LD case, iteration $(k+1)") 
				xtemp = (xk + yk) / 2
                nrmT=norm(xtemp)
				nrmG=norm(xtemp)
				xk=xtemp
				break
			else
     			Awk  = A*wk
				Ark  = wk - tk*Awk
				rtAr = rk'*Ark
				gtAr = gk'*Ark
				delta = rtAr*gtw - gtAr^2
				rtg = rk'*gk
				alpha = (rtg*gtAr - ng2*rtAr)/delta
				beta  = (-rtg*gtw + ng2*gtAr)/delta
				xtemp=xk+alpha*gk + beta*rk
				nrmT=norm(xtemp)
				gtemp=A*xtemp-b
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
				mu = (sk'*gtemp)/(sk'*gsk)
				xk = (1-mu)*xtemp+mu*xprev
				gk = (1-mu)*gtemp+mu*gprev
				xprev.=xaux
				gprev.=gaux
			end	
		else
			xk .= yk
			nrmG = nrmR
			k=k+1
			break
		end
		nrmG = norm(gk)
		k = k + 1
		if k >= mxitr
			break
		end
	end

	elapsed = time() - start_time

	return xk, k, elapsed, nrmG, f(xk, A, b)

end


