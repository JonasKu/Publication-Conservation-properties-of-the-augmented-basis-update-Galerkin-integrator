using Base: Float64
include("settings.jl")
include("SolverDLRA.jl")

using PyPlot
using DelimitedFiles

s = Settings(51,51,50); # create settings class with 251 x 251 spatial cells and a maximal rank of 100

################################################################
######################### execute code #########################
################################################################

################### run explicit Euler ###################
s.ϑ = 5e-2;
solver = SolverDLRA(s);
@time u1 = SolveUnconventional(solver);
u1 = Vec2Mat(s.NCellsX,s.NCellsY,u1)

############################################################
######################### plotting #########################
############################################################

################### read in reference solution ###################
lsRef = readdlm("exactLineSource.txt", ',', Float64);
xRef = lsRef[:,1];
phiRef = lsRef[:,2];
lsRefFull = readdlm("refPhiFull.txt", ',', Float64);

################### plot reference cut ###################
fig = figure("u cut ref",figsize=(10, 10), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
ax.plot(xRef,phiRef, "k-", linewidth=2, label="exact", alpha=0.8)
ylabel("scalar flux", fontsize=20)
xlabel(L"$x$", fontsize=20)
ax.set_xlim([s.a,s.b])
ax.tick_params("both",labelsize=20) 
tight_layout()
show()
savefig("scalar_flux_reference_cut.pdf")

################### plot reference full ###################
fig = figure("scalar_flux_reference",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid,s.yMid, lsRefFull,vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title("scalar flux, reference", fontsize=25)
savefig("scalar_flux_reference.pdf")

################### plot scalar fluxes full ###################

## explicit Euler
fig = figure("scalar_flux",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid,s.yMid, 4.0*pi*sqrt(2)*u1[:,:,1],vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"scalar flux, full", fontsize=25)
show()
savefig("scalar_flux_exp_Euler_nx$(s.NCellsX).png")


################### plot scalar fluxes cut ###################

fig = figure("u cut",figsize=(10, 10), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
ax.plot(xRef,phiRef, "k-", linewidth=2, label="exact", alpha=0.8)
ax.plot(s.yMid,4.0*pi*sqrt(2)*u1[Int(floor(s.NCellsX/2+1)),:,1], "b--", linewidth=2, label=L"DLRA", alpha=0.8)
ax.legend(loc="upper left", fontsize=20)
ylabel("scalar flux", fontsize=20)
xlabel(L"$x$", fontsize=20)
ax.set_xlim([s.a,s.b])
ax.set_ylim([-0.01,0.5])
ax.tick_params("both",labelsize=20) 
tight_layout()
show()
savefig("u_cut_exp_Euler_nx$(s.NCellsX).pdf")

println("main finished")
