
include("util.jl")
include("ellip_center.jl")
include("relaxed_ellip_center.jl")
include("ellip_center_with_momentum.jl")
include("conjugate_gradient.jl")
include("DWGM.jl")
include("BB1.jl")
include("ABBmin1.jl")

using ImageView
using MAT
using Printf
using FileIO
using Images

function run_solvers_random_convex(n, ncond, Nc)

	mxitr = 500000
	tol   = 1e-7
	vec1  = zeros(3)
	vec2  = zeros(3)
	vec3  = zeros(3)
	vec4  = zeros(3)
	vec5  = zeros(3)
	vec6  = zeros(3)
	vec7  = zeros(3)

	for i in 1:Nc

		A, b, x0 = sintetic_data(n, ncond)

		println("Running solvers for n = $n, ncond = $ncond, iteration = $i")

		x, elapsed, optimal_value, k = ellip_center!(A, b, tol, x0, mxitr)
		vec1[1] += k
		vec1[2]+=elapsed
		vec1[3]+=norm(b-A*x)

		omega = 0.9
		xk, elapsed, nrmG, k, opt_val = relaxed_ellip_center(A, x0, b, mxitr, tol, omega)
		vec2[1] += k
		vec2[2] += elapsed
		vec2[3] += norm(b - A * xk)

		xk, k, elapsed, nrmG, opt_val = ellip_center_with_momentum(A, x0, b, mxitr, tol)
		vec3[1] += k
		vec3[2] += elapsed
		vec3[3] += nrmG

		x, elapsed, optimal_value, k = conjugate_gradient(A, b, x0, tol)
		vec4[1] += k
		vec4[2] += elapsed
		vec4[3] += norm(b-A*x)


		xk, k, tf, res = DWGM(A, x0, b, mxitr, tol)
		vec5[1] += k
		vec5[2] += tf
		vec5[3] += res


		x, elapsed, optimal_value, k = barzilai_borwein(A, b, tol, x0, true)
		vec6[1] += k
		vec6[2] += elapsed
		vec6[3] += optimal_value

		m = 9
		xk, k, tf, res = ABBmin1(A, x0, b, mxitr, tol, m)
		vec7[1] += k
		vec7[2] += tf
		vec7[3] += res

	end
	vec1./=Nc
	vec2./=Nc
	vec3./=Nc
	vec4./=Nc
	vec5./=Nc
	vec6./=Nc
	vec7./=Nc

	return vec1, vec2, vec3, vec4, vec5, vec6, vec7

end



function run_solvers_experiment_2(n, Nc)

	mxitr = 500000
	tol   = 1e-7
	vec1  = zeros(3)
	vec2  = zeros(3)
	vec3  = zeros(3)
	vec4  = zeros(3)
	vec5  = zeros(3)
	vec6  = zeros(3)
	vec7  = zeros(3)

	for i in 1:Nc

		A, b, x0 = two_point_boundary_value(n)

		println("Running solvers for n = $n, iteration = $i")

		x, elapsed, optimal_value, k = ellip_center!(A, b, tol, x0, mxitr)
		vec1[1] += k
		vec1[2]+=elapsed
		vec1[3]+=norm(b-A*x)

		omega = 0.9
		xk, elapsed, nrmG, k, opt_val = relaxed_ellip_center(A, x0, b, mxitr, tol, omega)
		vec2[1] += k
		vec2[2] += elapsed
		vec2[3] += norm(b - A * xk)

		xk, k, elapsed, nrmG, opt_val = ellip_center_with_momentum(A, x0, b, mxitr, tol)
		vec3[1] += k
		vec3[2] += elapsed
		vec3[3] += nrmG

		x, elapsed, optimal_value, k = conjugate_gradient(A, b, x0, tol)
		vec4[1] += k
		vec4[2] += elapsed
		vec4[3] += norm(b-A*x)


		xk, k, tf, res = DWGM(A, x0, b, mxitr, tol)
		vec5[1] += k
		vec5[2] += tf
		vec5[3] += res


		x, elapsed, optimal_value, k = barzilai_borwein(A, b, tol, x0, true)
		vec6[1] += k
		vec6[2] += elapsed
		vec6[3] += optimal_value

		m = 9
		xk, k, tf, res = ABBmin1(A, x0, b, mxitr, tol, m)
		vec7[1] += k
		vec7[2] += tf
		vec7[3] += res

	end
	vec1./=Nc
	vec2./=Nc
	vec3./=Nc
	vec4./=Nc
	vec5./=Nc
	vec6./=Nc
	vec7./=Nc

	return vec1, vec2, vec3, vec4, vec5, vec6, vec7

end

function run_solvers_experiment_3(A, Nc)

	mxitr = 500000
	tol   = 1e-7
	vec1  = zeros(3)
	vec2  = zeros(3)
	vec3  = zeros(3)
	vec4  = zeros(3)
	vec5  = zeros(3)
	vec6  = zeros(3)
	vec7  = zeros(3)

	for i in 1:Nc

		n, _ = size(A);

		x0 = ones(n);
		for i in 1:n
			if mod(i, 2) == 0
				x0[i] = -1;
			end
		end
		x_opt = collect(1:n)
		x_opt = (1/n)*x_opt
		b = A*x_opt

		println("Running solvers for iteration = $i")

		x, elapsed, optimal_value, k = ellip_center!(A, b, tol, x0, mxitr)
		vec1[1] += k
		vec1[2]+=elapsed
		vec1[3]+=norm(b-A*x)

		omega = 0.9
		xk, elapsed, nrmG, k, opt_val = relaxed_ellip_center(A, x0, b, mxitr, tol, omega)
		vec2[1] += k
		vec2[2] += elapsed
		vec2[3] += norm(b - A * xk)

		xk, k, elapsed, nrmG, opt_val = ellip_center_with_momentum(A, x0, b, mxitr, tol)
		vec3[1] += k
		vec3[2] += elapsed
		vec3[3] += nrmG

		x, elapsed, optimal_value, k = conjugate_gradient(A, b, x0, tol)
		vec4[1] += k
		vec4[2] += elapsed
		vec4[3] += norm(b-A*x)


		xk, k, tf, res = DWGM(A, x0, b, mxitr, tol)
		vec5[1] += k
		vec5[2] += tf
		vec5[3] += res


		x, elapsed, optimal_value, k = barzilai_borwein(A, b, tol, x0, true)
		vec6[1] += k
		vec6[2] += elapsed
		vec6[3] += norm(b-A*x)

		m = 9
		xk, k, tf, res = ABBmin1(A, x0, b, mxitr, tol, m)
		vec7[1] += k
		vec7[2] += tf
		vec7[3] += res

	end
	vec1./=Nc
	vec2./=Nc
	vec3./=Nc
	vec4./=Nc
	vec5./=Nc
	vec6./=Nc
	vec7./=Nc

	return vec1, vec2, vec3, vec4, vec5, vec6, vec7

end


function test_random_convex(dim, kappa, Nc)
	open("experiment_1.txt", "w") do f
		for i ∈ 1:length(dim)
			n = dim[i]
			for j ∈ 1:length(kappa)
				ncond = kappa[j]
				vec1, vec2, vec3, vec4, vec5, vec6, vec7 = run_solvers_random_convex(n, ncond, Nc)
				write(f, "n = $n\t  ncond=$ncond\n")
				write(f, "ME\tIter= $(@sprintf("%.4f", vec1[1])) \t Time= $(@sprintf("%.4f", vec1[2]))\t NrmG= $(@sprintf("%.4f", vec1[3]))\n")
				write(f, "REM\t Iter= $(@sprintf("%.4f", vec2[1]))\t Time= $(@sprintf("%.4f", vec2[2]))\t NrmG= $(@sprintf("%.4f", vec2[3]))\n")
				write(f, "MEM\t Iter= $(@sprintf("%.4f", vec3[1]))\t Time= $(@sprintf("%.4f", vec3[2]))\t NrmG= $(@sprintf("%.4f", vec3[3]))\n")
				write(f, "CG\tIter= $(@sprintf("%.4f", vec4[1]))\t Time= $(@sprintf("%.4f", vec4[2]))\t NrmG= $(@sprintf("%.4f", vec4[3]))\n")
				write(f, "DWGM\tIter= $(@sprintf("%.4f", vec5[1]))\t Time= $(@sprintf("%.4f", vec5[2]))\t NrmG= $(@sprintf("%.4f", vec5[3]))\n")
				write(f, "BB1\tIter= $(@sprintf("%.4f", vec6[1]))\t Time= $(@sprintf("%.4f", vec6[2]))\t NrmG= $(@sprintf("%.4f", vec6[3]))\n")
				write(f, "ABBmin1\tIter= $(@sprintf("%.4f", vec7[1]))\t Time= $(@sprintf("%.4f", vec7[2]))\t NrmG= $(@sprintf("%.4f", vec7[3]))\n")
			end
		end
	end
end

function test_experiment_2(ns, Nc)
	open("experiment_2.txt", "w") do f
		for i ∈ 1:length(ns)
			n = ns[i]
			vec1, vec2, vec3, vec4, vec5, vec6, vec7 = run_solvers_experiment_2(n, Nc)
			write(f, "n = $n\n")
			write(f, "ME\tIter= $(@sprintf("%.4f", vec1[1])) \t Time= $(@sprintf("%.7f", vec1[2]))\t NrmG= $(@sprintf("%.4f", vec1[3]))\n")
			write(f, "REM\t Iter= $(@sprintf("%.4f", vec2[1]))\t Time= $(@sprintf("%.7f", vec2[2]))\t NrmG= $(@sprintf("%.4f", vec2[3]))\n")
			write(f, "MEM\t Iter= $(@sprintf("%.4f", vec3[1]))\t Time= $(@sprintf("%.7f", vec3[2]))\t NrmG= $(@sprintf("%.4f", vec3[3]))\n")
			write(f, "CG\tIter= $(@sprintf("%.4f", vec4[1]))\t Time= $(@sprintf("%.7f", vec4[2]))\t NrmG= $(@sprintf("%.4f", vec4[3]))\n")
			write(f, "DWGM\tIter= $(@sprintf("%.4f", vec5[1]))\t Time= $(@sprintf("%.7f", vec5[2]))\t NrmG= $(@sprintf("%.4f", vec5[3]))\n")
			write(f, "BB1\tIter= $(@sprintf("%.4f", vec6[1]))\t Time= $(@sprintf("%.7f", vec6[2]))\t NrmG= $(@sprintf("%.4f", vec6[3]))\n")
			write(f, "ABBmin1\tIter= $(@sprintf("%.4f", vec7[1]))\t Time= $(@sprintf("%.7f", vec7[2]))\t NrmG= $(@sprintf("%.4f", vec7[3]))\n")
		end
	end
end

function test_experiment_3(Nc)
	f=open("experiment_3.txt", "w")
	pasta = "C:\\Users\\vince\\Dropbox\\Softwares\\Relaxed_ME\\MATLAB\\Experiments\\datasets"
	for arquivo in readdir(pasta, join = true)
		if endswith(arquivo, ".mat")
			A = Array(matread(arquivo)["Problem"]["A"])
			println("Arquivo: $arquivo")
			vec1, vec2, vec3, vec4, vec5, vec6, vec7 = run_solvers_experiment_3(A, Nc)
			write(f, "Arquivo: $arquivo\n")
			write(f, "ME\tIter= $(@sprintf("%.4f", vec1[1])) \t Time= $(@sprintf("%.7f", vec1[2]))\t NrmG= $(@sprintf("%.4f", vec1[3]))\n")
			write(f, "RelaxME\t Iter= $(@sprintf("%.4f", vec2[1]))\t Time= $(@sprintf("%.7f", vec2[2]))\t NrmG= $(@sprintf("%.4f", vec2[3]))\n")
			write(f, "MomME\t Iter= $(@sprintf("%.4f", vec3[1]))\t Time= $(@sprintf("%.7f", vec3[2]))\t NrmG= $(@sprintf("%.4f", vec3[3]))\n")
			write(f, "CG\tIter= $(@sprintf("%.4f", vec4[1]))\t Time= $(@sprintf("%.7f", vec4[2]))\t NrmG= $(@sprintf("%.4f", vec4[3]))\n")
			write(f, "DWGM\tIter= $(@sprintf("%.4f", vec5[1]))\t Time= $(@sprintf("%.7f", vec5[2]))\t NrmG= $(@sprintf("%.4f", vec5[3]))\n")
			write(f, "BB1\tIter= $(@sprintf("%.4f", vec6[1]))\t Time= $(@sprintf("%.7f", vec6[2]))\t NrmG= $(@sprintf("%.4f", vec6[3]))\n")
			write(f, "ABBmin1\tIter= $(@sprintf("%.4f", vec7[1]))\t Time= $(@sprintf("%.7f", vec7[2]))\t NrmG= $(@sprintf("%.4f", vec7[3]))\n")
			write(f, "\n")
			flush(f)
		end
	end
end

function test_experiment_4()
	f=open("experiment_4.txt", "w")
	pasta = "C:\\Users\\vince\\Dropbox\\Softwares\\Relaxed_ME\\MATLAB\\Experiments\\images"
	for arquivo in readdir(pasta, join = true)
		if endswith(arquivo, ".tiff")
			img = load(arquivo)
			gray = Gray.(img)
			y = Float64.(gray)
			n, m = size(y)
			x0 = zeros(n, m)
			lambda = 0.1
			mxitr = 500000
			tol   = 1e-7
			println("Arquivo: $arquivo")
			
			xk, tf, nrmG, k = BB1_IS(x0, y, lambda, mxitr, tol)
			write(f, "Arquivo: $arquivo\n")

			write(f, "BB1_IS\tIter= $(@sprintf("%.4f", k)) \t Time= $(@sprintf("%.7f", tf))\t NrmG= $(@sprintf("%.4f", nrmG))\n")
			write(f, "\n")
			
			xk, tf, nrmG, k = MEM_IS(x0,y,lambda,mxitr,tol)
			write(f, "MEM_IS\tIter= $(@sprintf("%.4f", k)) \t Time= $(@sprintf("%.7f", tf))\t NrmG= $(@sprintf("%.4f", nrmG))\n")
			write(f, "\n")

			xk, tf, nrmG, k = ABBmin1_IS(x0,y,lambda,mxitr,tol,9)
			write(f, "ABBmin1_IS\tIter= $(@sprintf("%.4f", k)) \t Time= $(@sprintf("%.7f", tf))\t NrmG= $(@sprintf("%.4f", nrmG))\n")
			write(f, "\n")
  
			omega=0.9
            xk, tf, nrmG, k = ellip_center_IS(xk,y,lambda,mxitr,tol,omega)
			write(f, "ellip_center_Relaxed\tIter= $(@sprintf("%.4f", k)) \t Time= $(@sprintf("%.7f", tf))\t NrmG= $(@sprintf("%.4f", nrmG))\n")
			write(f, "\n")

			omega=1
            xk, tf, nrmG, k = ellip_center_IS(xk,y,lambda,mxitr,tol,omega)
			write(f, "ellip_center_IS\tIter= $(@sprintf("%.4f", k)) \t Time= $(@sprintf("%.7f", tf))\t NrmG= $(@sprintf("%.4f", nrmG))\n")
			write(f, "\n")

			flush(f)
		end
	end
end

function test()

	 dim   = [1000, 2500, 5000]
	 kappa = [3.0, 6.0, 9.0, 12.0]
	 Nc = 1
	 test_random_convex(dim, kappa, Nc)

	#  ns=[100, 200, 300, 400, 500, 600]
	#  test_experiment_2(ns, Nc)

	# img = load("C:\\Users\\vince\\Dropbox\\Softwares\\Relaxed_ME\\MATLAB\\Experiments\\images\\boat.tiff")
	# gray = Gray.(img)
	# imshow(gray)

	# test_experiment_3(Nc)

	test_experiment_4()

end

test()
