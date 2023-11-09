using Base: Float64
include("utils.jl")
include("settings.jl")
include("SolverDLRA.jl")

using PyPlot
using DelimitedFiles

nₚₙ = 31;     # nₚₙ = 39;
ϑₚ = 5e-1;    # parallel tolerance
ϑᵤ = 0.05;    # (unconventional) BUG tolerance
#ϑₚ₁ = 2.5e-2;      # parallel tolerance
#ϑᵤ₁ = 2.5e-1;    # (unconventional) BUG tolerance
#ϑₚ₂ = 1e-2;      # parallel tolerance
#ϑᵤ₂ = 1e-1;    # (unconventional) BUG tolerance

s = Settings(251, 251, nₚₙ, 100); # create settings class with 251 x 251 spatial cells and a maximal rank of 100
#s = Settings(50, 50, nₚₙ, 100); # create settings class with 251 x 251 spatial cells and a maximal rank of 100

################################################################
######################### execute code #########################
################################################################

################### run full method ###################
#solver = SolverDLRA(s);
#@time rhoFull = Solve(solver);
#Φ = Vec2Mat(s.NCellsX,s.NCellsY,rhoFull)

##################### low accuracy #####################
##################### higher accuracy #####################

################### run method Lukas ###################
s.ϑ = ϑᵤ;
solver = SolverDLRA(s);
@time ψ, rankInTimeₗ, massInTimeₗ = SolveBUGLukas(solver);
Φₗ = Vec2Mat(s.NCellsX, s.NCellsY, ψ);

@time ψ, rankInTime_n, massInTime_n = SolveBUGLukasNabla(solver);
Φ_n = Vec2Mat(s.NCellsX, s.NCellsY, ψ);

################### run BUG adaptive ###################
s.ϑ = ϑᵤ;
solver = SolverDLRA(s);
@time ψ, rankInTimeCᵣ, massInTimeCᵣ = SolveBUGAdaptive(solver);
ΦCᵣ = Vec2Mat(s.NCellsX, s.NCellsY, ψ);

s.ϑ = ϑᵤ;
solver = SolverDLRA(s);
@time ψ, rankInTimeᵣ, massInTimeᵣ = SolveBUGAdaptive_noncons(solver);
Φᵣ = Vec2Mat(s.NCellsX, s.NCellsY, ψ);


############################################################
######################### plotting #########################
############################################################
max_val = 0.45

################### read in reference solution ###################
lsRef = readdlm("exactLineSource.txt", ',', Float64);
xRef = lsRef[:, 1];
phiRef = lsRef[:, 2];
lsRefFull = readdlm("refPhiFull.txt", ',', Float64);

nxRef = size(lsRefFull, 1);
nyRef = size(lsRefFull, 2);
xRefFull = collect(range(s.a, s.b, nxRef));
yRefFull = collect(range(s.c, s.d, nxRef));

################### plot reference cut ###################
fig = figure(figsize=(10, 10), dpi=100)
ax = gca()
ax.plot(xRef, phiRef, "k-", linewidth=2, label="exact", alpha=0.8)
ylabel(L"\Phi", fontsize=20)
xlabel(L"$x$", fontsize=20)
ax.set_xlim([s.a, s.b])
ax.tick_params("both", labelsize=20)
ax.grid(true)
tight_layout()
#show()
savefig("results/scalar_flux_reference_cut.pdf")
savefig("results/scalar_flux_reference_cut.png")

################### plot reference full ###################
fig = figure(figsize=(10, 10), dpi=100)
ax = gca()
pcolormesh(xRefFull, yRefFull, lsRefFull, vmin=0.0, vmax=max_val)
ax.tick_params("both", labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.tight_layout()

savefig("results/scalar_flux_reference.png")
savefig("results/scalar_flux_reference.pdf")
#
################### plot scalar fluxes full ###################

############################################################
######################### plotting #########################
############################################################
X = (s.xMid[2:end-1]' .* ones(size(s.xMid[2:end-1])));
Y = (s.yMid[2:end-1]' .* ones(size(s.yMid[2:end-1])))';
Φ = Φᵣ
## full
maxV = maximum(Φ[2:end-1, 2:end-1])
minV = max(1e-7, minimum(4.0 * pi * sqrt(2) * Φ[2:end-1, 2:end-1]))

fig = figure(figsize=(10, 10), dpi=100)
#ax = gca()
pcolormesh(X, Y, 4.0 * pi * sqrt(2) * Φ[2:(end-1), (end-1):-1:2]', vmin=0.0, vmax=maximum(max_val))
#ax.tick_params("both", labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.tight_layout()
#show()
plt.savefig("results/scalar_flux_PN_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")
plt.savefig("results/scalar_flux_PN_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).pdf")
#plt.clc()
## DLRA BUG adaptive
maxV = maximum(Φᵣ[2:end-1, 2:end-1])
minV = max(1e-7, minimum(4.0 * pi * sqrt(2) * Φᵣ[2:end-1, 2:end-1]))
fig = figure(figsize=(10, 10), dpi=100)
ax = gca()
pcolormesh(X, Y, 4.0 * pi * sqrt(2) * Φᵣ[2:(end-1), (end-1):-1:2]', vmin=0.0, vmax=max_val)
ax.tick_params("both", labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.tight_layout()
#show()
plt.savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")
plt.savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).pdf")
## DLRA cont. cons. nabla
maxV = maximum(Φ_n[2:end-1, 2:end-1])
minV = max(1e-7, minimum(4.0 * pi * sqrt(2) * Φᵣ[2:end-1, 2:end-1]))
fig = figure(figsize=(10, 10), dpi=100)
ax = gca()
pcolormesh(X, Y, 4.0 * pi * sqrt(2) * Φ_n[2:(end-1), (end-1):-1:2]', vmin=0.0, vmax=max_val)
ax.tick_params("both", labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.tight_layout()
#show()
plt.savefig("results/scalar_flux_DLRA_Lukas_nabla_X_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")
plt.savefig("results/scalar_flux_DLRA_Lukas_nabla_X_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).pdf")
#plt.clc()
## DLRA cont. cons.
maxV = maximum(Φₗ[2:end-1, 2:end-1])
minV = max(1e-7, minimum(4.0 * pi * sqrt(2) * Φᵣ[2:end-1, 2:end-1]))
fig = figure(figsize=(10, 10), dpi=100)
ax = gca()
pcolormesh(X, Y, 4.0 * pi * sqrt(2) * Φₗ[2:(end-1), (end-1):-1:2]', vmin=0.0, vmax=max_val)
ax.tick_params("both", labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.tight_layout()
#show()
plt.savefig("results/scalar_flux_DLRA_Lukas_no_nabla_X_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")
plt.savefig("results/scalar_flux_DLRA_Lukas_no_nabla_X_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).pdf")
#plt.clc()
maxV = maximum(ΦCᵣ[2:end-1, 2:end-1])
minV = max(1e-7, minimum(4.0 * pi * sqrt(2) * ΦCᵣ[2:end-1, 2:end-1]))
fig = figure(figsize=(10, 10), dpi=100)
ax = gca()
pcolormesh(X, Y, 4.0 * pi * sqrt(2) * ΦCᵣ[2:(end-1), (end-1):-1:2]', vmin=0.0, vmax=max_val)
ax.tick_params("both", labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.tight_layout()
#show()
plt.savefig("results/scalar_flux_DLRA_cons_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")
plt.savefig("results/scalar_flux_DLRA_cons_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).pdf")


################### plot scalar fluxes cut ###################

fig = figure()#figsize=(8, 6))#("Φ cut, BUG", figsize=(10, 10), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
ax.plot(xRef, phiRef, "k-", linewidth=2, label="exact", alpha=1.0)
ax.plot(s.yMid, 4.0 * pi * sqrt(2) * Φₗ[Int(floor(s.NCellsX / 2 + 1)), :], "b-.", linewidth=2, label="cont. cons. BUG, no \$\\nabla X_0\$", alpha=0.4)
ax.plot(s.yMid, 4.0 * pi * sqrt(2) * Φ_n[Int(floor(s.NCellsX / 2 + 1)), :], "g--", linewidth=2, label=label="cont. cons. BUG, \$\\nabla X_0\$", alpha=0.4)
ax.plot(s.yMid, 4.0 * pi * sqrt(2) * ΦCᵣ[Int(floor(s.NCellsX / 2 + 1)), :], "m:", linewidth=2, label="cons. aug. BUG", alpha=0.9)
ax.plot(s.yMid, 4.0 * pi * sqrt(2) * Φᵣ[Int(floor(s.NCellsX / 2 + 1)), :], "y-", linewidth=2, label="aug. BUG", alpha=0.8)
ax.legend(loc="lower center", fontsize=13)
plt.ylabel(L"\Phi", fontsize=13)
plt.xlabel(L"$x$", fontsize=13)
ax.set_xlim([s.a, s.b])
ax.set_ylim([-0.01, 0.45])
ax.tick_params("both", labelsize=14)
ax.grid(true)

tight_layout()
#show()
savefig("results/phi_cut_BUGnx$(s.NCellsX).png")
savefig("results/phi_cut_BUGnx$(s.NCellsX).pdf")
plt.close()

# plot rank in time
fig = figure()#figsize=(8, 6))#("ranks", figsize=(10, 10), dpi=100)
ax = gca()
ax.plot(rankInTimeₗ[1, :], rankInTimeₗ[2, :], "b--", label="cont. cons. BUG, no \$\\nabla X_0\$", linewidth=2, alpha=0.4)
ax.plot(rankInTime_n[1, :], rankInTime_n[2, :], "g--", linewidth=2, label="cont. cons. BUG, \$\\nabla X_0\$", alpha=0.4)
ax.plot(rankInTimeCᵣ[1, :], rankInTimeCᵣ[2, :], "m--", linewidth=2, label="cons. aug. BUG", alpha=0.9)
ax.plot(rankInTimeᵣ[1, :], rankInTimeᵣ[2, :], "y-", linewidth=2, label="aug. BUG", alpha=0.8)

#ax.set_xlim([rankInTimeᵣ[1, 1], rankInTimeᵣ[1, end]])
#ax.set_ylim([0, max(maximum(rankInTimeᵣ[2, 2:end]), maximum(rankInTimeₗ[2, 2:end])) + 2])
ax.set_xlabel("time", fontsize=13);
ax.set_ylabel("rank", fontsize=13);
ax.tick_params("both", labelsize=14);
ax.legend(loc="upper left", fontsize=13);
plt.xlim([0, 1])
plt.ylim([0, 175])
#fig.canvas.draw() # Update the figure
ax.grid(true)
tight_layout()

PyPlot.savefig("results/ranks_linesource.png")
PyPlot.savefig("results/ranks_linesource.pdf")
plt.close()


# plot mass in time
fig = figure()#("mass", figsize=(15, 12), dpi=100)
ax = gca()
#ax.plot(massInTimeₗ[1, :], abs.(massInTimeₗ[2, :] .- massInTimeₗ[2, 1]) / 1e5, "r--", label="cont. cons. BUG", linewidth=2, alpha=1.0)
ax.plot(massInTimeCᵣ[1, :], abs.(massInTimeCᵣ[2, :] .- massInTimeCᵣ[2, 1]) / 1e5, "m--", label="cons. aug. BUG", linewidth=2, alpha=1.0)
ax.plot(massInTimeᵣ[1, :], abs.(massInTimeᵣ[2, :] .- massInTimeᵣ[2, 1]) / 1e5, "y--", label="aug. BUG", linewidth=2, alpha=0.8)
ax.set_xlim([massInTimeᵣ[1, 1], massInTimeᵣ[1, end]])
ax.set_xlabel("time", fontsize=20);
ax.set_ylabel("error mass", fontsize=20);
ax.tick_params("both", labelsize=20);
ax.legend(loc="upper left", fontsize=20);
#ax.yaxis.offsetText.set_fontsize(25)
ax.grid(true)
ax.yaxis.offsetText.set_fontsize(15)
tight_layout()
PyPlot.savefig("results/mass_err_cont_cons_$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")
PyPlot.savefig("results/mass_err_cont_cons_$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).pdf")
plt.close()
writedlm("results/ranks_$(s.problem)_thetaBUG$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", rankInTimeᵣ)
writedlm("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", Φᵣ)

plt.close()

# plot mass in time
fig = figure()#("mass", figsize=(15, 12), dpi=100)
ax = gca()
ax.plot(massInTimeₗ[1, :], abs.(massInTimeₗ[2, :] .- massInTimeₗ[2, 1]) / 1e5, "b--", label="cont. cons. BUG, no \$\\nabla X_0\$", linewidth=2, alpha=1.0)
ax.plot(massInTimeCᵣ[1, :], abs.(massInTimeCᵣ[2, :] .- massInTimeCᵣ[2, 1]) / 1e5, "m--", label="cons. aug. BUG", linewidth=2, alpha=1.0)
ax.set_xlim([massInTimeᵣ[1, 1], massInTimeᵣ[1, end]])
#ax.set_ylim([0,max(maximum(massInTimeᵣ[2,2:end]),maximum(massInTimeᵣ[2,2:end]))+2])
ax.set_xlabel("time", fontsize=20);
ax.set_ylabel("error mass", fontsize=20);
ax.tick_params("both", labelsize=20);
ax.legend(loc="upper left", fontsize=20);
#ax.yaxis.offsetText.set_fontsize(25)
ax.grid(true)
ax.yaxis.offsetText.set_fontsize(15)
tight_layout()
PyPlot.savefig("results/mass_err_cont_cons_$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ)_1.png")
PyPlot.savefig("results/mass_err_cont_cons_$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ)_1.pdf")
plt.close()

# plot mass in time
fig = figure()#("mass", figsize=(15, 12), dpi=100)
ax = gca()
ax.plot(massInTimeCᵣ[1, :], abs.(massInTimeCᵣ[2, :] .- massInTimeCᵣ[2, 1]) / 1e5, "m--", label="cons. aug. BUG, no \$\\nabla X_0\$", linewidth=2, alpha=1.0)
ax.plot(massInTimeₗ[1, :], abs.(massInTime_n[2, :] .- massInTime_n[2, 1]) / 1e1 / 1e5, "g--", label="cont. cons. BUG, \$\\nabla X_0\$", linewidth=2, alpha=1.0)
ax.set_xlim([massInTimeCᵣ[1, 1], massInTimeCᵣ[1, end]])
#ax.set_ylim([0,max(maximum(massInTimeᵣ[2,2:end]),maximum(massInTimeᵣ[2,2:end]))+2])
ax.set_xlabel("time", fontsize=20);
ax.set_ylabel("error mass", fontsize=20);
ax.tick_params("both", labelsize=20);
ax.legend(loc="upper left", fontsize=20);
#ax.yaxis.offsetText.set_fontsize(25)
fig.canvas.draw() # Update the figure
ax.grid(true)
ax.yaxis.offsetText.set_fontsize(15)
tight_layout()
PyPlot.savefig("results/mass_err_cont_cons_adv$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")
PyPlot.savefig("results/mass_err_cont_cons_adv$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).pdf")
plt.close()
writedlm("results/ranks_$(s.problem)_thetaBUG$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", rankInTimeᵣ)
writedlm("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", Φᵣ)

println("main finished")

