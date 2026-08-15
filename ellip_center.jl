

include("util.jl")

function ellip_center!(A::AbstractMatrix,b::Vector,tol::Float64,xk::Vector,mxitr::Int)

	gk = A*xk - b
	start_time = time()
	k = 0
	tol_ME=1e-7
	nrmG = norm(gk)
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
            if abs(nrmR-nrmG)<tol_ME*max(1,abs(nrmR))
			# if abs(nrmR-nrmG)<tol_ME 
                println("Entering the LD case, iteration $(k+1)")
				xk = (xk + yk) / 2
                # nrmG=0
				nrmG=norm(xk)
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
				xk=xk+alpha*gk + beta*rk
				gk=A*xk-b
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

	return xk, elapsed,f(xk, A, b), nrmG, k
end


