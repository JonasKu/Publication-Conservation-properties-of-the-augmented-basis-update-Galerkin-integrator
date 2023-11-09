using Base: Float64
include("utils.jl")
include("settings.jl")
include("SolverDLRA.jl")

using PyPlot
using DelimitedFiles

close("all")

nₚₙ = 21;
ϑₚ = 2e-2;      # parallel tolerance
ϑᵤ = 5e-2;    # (unconventional) BUG tolerance # BUG 0.05, parrallel 0.02 looks okay
ϑₚₕ = 1e-2;      # parallel tolerance
ϑᵤₕ = 3e-2;    # (unconventional) BUG tolerance

s = Settings(351,351, nₚₙ, 50,"Lattice"); # create settings class with 351 x 351 spatial cells and a rank of 50

################################################################
######################### execute code #########################
################################################################

##################### classical checkerboard #####################

################### run full method ###################
solver = SolverDLRA(s);
@time rhoFull = Solve(solver);
rhoFull = Vec2Mat(s.NCellsX,s.NCellsY,rhoFull)

##################### low tolerance #####################

################### run BUG adaptive ###################
s.ϑ = ϑᵤ;
solver = SolverDLRA(s);
@time rhoDLRA,rankInTime,etaBUG = SolveBUGAdaptiveRejection(solver);
rhoDLRA = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRA);

################### run parallel ###################
s.ϑ = ϑₚ;
solver = SolverDLRA(s);
@time rhoDLRAp,rankInTimep, eta = SolveParallelRejection(solver);
rhoDLRAp = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRAp);

##################### high tolerance #####################

################### run BUG adaptive ###################
s.ϑ = ϑᵤₕ;
solver = SolverDLRA(s);
@time rhoDLRAₕ,rankInTimeₕ,etaBUGₕ = SolveBUGAdaptiveRejection(solver);
rhoDLRAₕ = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRAₕ);

################### run parallel ###################
s.ϑ = ϑₚₕ;
solver = SolverDLRA(s);
@time rhoDLRApₕ,rankInTimepₕ, etaₕ = SolveParallelRejection(solver);
rhoDLRApₕ = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRApₕ);

############################################################
######################### plotting #########################
############################################################

X = (s.xMid[2:end-1]'.*ones(size(s.xMid[2:end-1])));
Y = (s.yMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))';

## full
maxV = maximum(rhoFull[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoFull[2:end-1,2:end-1]))
idxNeg = findall((rhoFull.<=0.0))
rhoFull[idxNeg] .= NaN;
fig = figure("full, log",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoFull[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, P$_{21}$", fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_PN_$(s.problem)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA BUG adaptive
maxV = maximum(rhoDLRA[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoDLRA[2:end-1,2:end-1]))
idxNeg = findall((rhoDLRA.<=0.0))
rhoDLRA[idxNeg] .= NaN;
fig = figure("BUG, log, ϑ coarse",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRA[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, BUG, $\vartheta =$ "*LaTeXString(string(ϑᵤ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel
maxV = maximum(rhoDLRAp[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoDLRAp[2:end-1,2:end-1]))
idxNeg = findall((rhoDLRAp.<=0.0))
rhoDLRAp[idxNeg] .= NaN;
fig = figure("parallel, log, ϑ coarse",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRAp[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel, $\vartheta =$ "*LaTeXString(string(ϑₚ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_$(s.problem)_theta$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA BUG adaptive
maxV = maximum(rhoDLRAₕ[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoDLRAₕ[2:end-1,2:end-1]))
idxNeg = findall((rhoDLRAₕ.<=0.0))
rhoDLRAₕ[idxNeg] .= NaN;
fig = figure("BUG, log",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRAₕ[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, BUG, $\vartheta =$ "*LaTeXString(string(ϑᵤₕ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤₕ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel
maxV = maximum(rhoDLRApₕ[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoDLRApₕ[2:end-1,2:end-1]))
idxNeg = findall((rhoDLRApₕ.<=0.0))
rhoDLRApₕ[idxNeg] .= NaN;
fig = figure("parallel, log",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRApₕ[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel, $\vartheta =$ "*LaTeXString(string(ϑₚₕ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_$(s.problem)_theta$(ϑₚₕ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

# plot rank in time
fig = figure("ranks",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(rankInTime[1,:],rankInTime[2,:], "k--", label=L"BUG, $\vartheta =$ "*LaTeXString(string(ϑᵤ)), linewidth=2, alpha=1.0)
ax.plot(rankInTimep[1,:],rankInTimep[2,:], "r--", label=L"parallel, $\vartheta =$ "*LaTeXString(string(ϑₚ)), linewidth=2, alpha=1.0)
ax.plot(rankInTimeₕ[1,:],rankInTimeₕ[2,:], "k:", label=L"BUG, $\vartheta =$ "*LaTeXString(string(ϑᵤₕ)), linewidth=2, alpha=1.0)
ax.plot(rankInTimepₕ[1,:],rankInTimepₕ[2,:], "r:", label=L"parallel, $\vartheta =$ "*LaTeXString(string(ϑₚₕ)), linewidth=2, alpha=1.0)
ax.set_xlim([rankInTime[1,1],rankInTime[1,end]+0.05])
ax.set_ylim([0,max(maximum(rankInTimeₕ[2,2:end]),maximum(rankInTimepₕ[2,2:end]))+2])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel("rank", fontsize=30);
ax.tick_params("both",labelsize=30) ;
ax.legend(loc="lower right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/ranks_$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

etaBoundᵤ = s.cη * ϑᵤ  / s.Δt
etaBoundₚ = s.cη * ϑₚ  / s.Δt
fig = figure("eta",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(etaBUG[:,1],etaBUG[:,2], "r:", label=L"BUG, $\vartheta =$ "*LaTeXString(string(ϑᵤ)), linewidth=2, alpha=1.0)
ax.plot(eta[:,1],eta[:,2], "b--", label=L"parallel, $\vartheta =$ "*LaTeXString(string(ϑₚ)), linewidth=2, alpha=1.0)
ax.plot(eta[:,1],etaBoundᵤ*ones(size(eta[:,1])), "r-", label=L"$c\vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(eta[:,1],etaBoundₚ*ones(size(eta[:,1])), "b-", label=L"$c\vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.set_xlim([eta[1,1],eta[end,1]+0.05])
ax.set_ylim([0,max(maximum(etaₕ[2:end,2]),maximum(eta[2:end,2]),etaBoundᵤ,etaBoundₚ)+1])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/eta_$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

etaBoundᵤₕ = s.cη * ϑᵤₕ  / s.Δt
etaBoundₚₕ = s.cη * ϑₚₕ  / s.Δt
fig = figure("eta fine",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(etaBUGₕ[:,1],etaBUGₕ[:,2], "r:", label=L"BUG, $\vartheta =$ "*LaTeXString(string(ϑᵤₕ)), linewidth=2, alpha=1.0)
ax.plot(etaₕ[:,1],etaₕ[:,2], "b--", label=L"parallel, $\vartheta =$ "*LaTeXString(string(ϑₚₕ)), linewidth=2, alpha=1.0)
ax.plot(etaBUGₕ[:,1],etaBoundᵤₕ*ones(size(etaBUGₕ[:,1])), "r-", label=L"$c\vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(etaₕ[:,1],etaBoundₚₕ*ones(size(etaₕ[:,1])), "b-", label=L"$c\vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.set_xlim([etaₕ[1,1],etaₕ[end,1]+0.05])
ax.set_ylim([0,max(maximum(etaₕ[2:end,2]),maximum(eta[2:end,2]),etaBoundᵤₕ,etaBoundₚₕ)+1])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/etafine_$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

fig = figure("setup",figsize=(10,10),dpi=100)
ax = fig.add_subplot(111)
rect1 = matplotlib.patches.Rectangle((1,5), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((5,5), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((4,4), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((2,4), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((3,3), 1.0, 1.0, color="orange")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((1,3), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((5,3), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((4,2), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((2,2), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((3,1), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((1,1), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((5,1), 1.0, 1.0, color="black")
ax.add_patch(rect1)
ax.grid()
plt.xlim([0, 7])
plt.ylim([0, 7])
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title("lattice testcase", fontsize=25)
tight_layout()
plt.show()
savefig("results/setup_lattice_testcase.png")


println("main finished")
