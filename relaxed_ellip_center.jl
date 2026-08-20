
include("util.jl")

function relaxed_ellip_center(A::AbstractMatrix, xk::Vector, b::Vector, mxitr::Int, tol::Float64, omega::Float64)

	gk = A*xk - b
	start_time = time()
	k = 0
	tol_ME=1e-7
	nrmG = norm(gk)
	xtemp=zeros(size(xk))
    gtemp=zeros(size(xk)) 

	while nrmG > tol 
		wk = A*gk
		gtw = gk'*wk
		ng2 = nrmG^2
		tk = 2*(ng2/gtw)
		yk = xk - tk*gk
		rk = gk - tk*wk
		nrmR = norm(rk)

		if nrmR > tol 
			if abs(nrmR-nrmG)<tol_ME*max(1,nrmR)
				println("Entering the LD case, iteration $(k+1)") 
				xtemp = (xk + yk) / 2
				gtemp=(gk+rk)/2
				nrmG=norm(gtemp)
				xk=xtemp
				k=k+1
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
				gtemp=gk+alpha*wk+beta*Ark
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
		if k >= mxitr
			break
		end
	end

	elapsed = time() - start_time

	return xk, elapsed, nrmG, k, f(xk, A, b)
end


