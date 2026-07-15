
include("util.jl")

function ellip_center_with_momentum(A::Matrix, xk::Vector, b::Vector, mxitr::Int, tol::Float64)
	# % ELLIP_CENTER Implementación del método de optimización en MATLAB
	# % Traducido de Julia.

	gk = A*xk - b
	start_time = time();
	k = 0
	nrmG = norm(gk)
	xold = xk
	gold = gk
	counter = 0
	while nrmG > tol 
		xk_prev = xk
		gk_prev = gk
		wk = A*gk
		gtw = gk'*wk
		ng2 = nrmG^2
		tk = 2*(ng2/gtw)
		yk = xk - tk*gk
		rk = gk - tk*wk
		nrmR = norm(rk)

		if nrmR > tol
			Awk  = A*wk
			Ark  = wk - tk*Awk
			rtAr = rk'*Ark
			gtAr = gk'*Ark

			delta = rtAr*gtw - gtAr^2
			if abs(delta) > 0
				rtg = rk'*gk

				alpha = (rtg*gtAr - ng2*rtAr)/delta
				beta  = (-rtg*gtw + ng2*gtAr)/delta

				xk_EM = xk + alpha*gk + beta*rk
				gk_EM = gk + alpha*wk + beta*Ark

			else
				counter = counter + 1
				xtemp = (xk + yk) / 2
				gtemp = 0.5*(gk + rk)
				xk_EM = (1-omega)*xk + omega*xtemp
				gk_EM = (1-omega)*gk + omega*gtemp
				if (norm(A*x_k_EM - b) < tol)
					break
					nrmG=0
				end
			end
		else
			xk = yk
			nrmG = nrmR
			break
		end

		sk = xold - xk_EM
		gsk = gold - gk_EM
		mu = -(sk'*gk_EM)/(sk'*gsk)
		xk = xk_EM + mu*sk
		gk = gk_EM + mu*gsk

		nrmG = norm(gk)
		xold = xk_prev
		gold = gk_prev
		k = k + 1
		if k > mxitr
			break;
		end
	end
	elapsed = time() - start_time

	return xk, k, elapsed, nrmG, f(xk, A, b)

end


