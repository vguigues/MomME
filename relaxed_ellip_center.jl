
include("util.jl")

function relaxed_ellip_center(A::Matrix, xk::Vector, b::Vector, mxitr::Int, tol::Float64, omega::Float64)

	gk = A*xk - b
	start_time = time()
	k = 0
	nrmG = norm(gk)
	counter = 0
	while nrmG > tol
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
			rtAr = rk'*Ark;
			gtAr = gk'*Ark;

			delta = rtAr*gtw - gtAr^2;
			if abs(delta) > 0
				rtg = rk'*gk;

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
			xk = yk
			nrmG = nrmR
			break
		end
		nrmG = norm(gk)
		k = k + 1

		if k > mxitr
			break
		end
	end

	elapsed = time() - start_time

	return xk, elapsed, nrmG, k, f(xk, A, b)
end


