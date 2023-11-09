include("settings.jl")
include("Solver.jl")

using DelimitedFiles
using PyPlot

close("all")

s = Settings();

############################
cfl = 0.4
solver = Solver(s);

@time tEnd, uBUG, ρBUG, rBUG = SolveBUG_cons_Lukas(solver); #cont. cons. with nabla X_0

@time tEnd, uBUGc, ρBUGc, rBUGc = SolveBUG_cont_cons(solver); #cont. cons. no nabla X_0

@time tEnd, uBUG_cons, ρ, rBUG_cons = SolveBUG_cons(solver); #cons. aug. BUG

@time tEnd, uBUG_non_cons, ρaug, rBUG_non_cons = SolveBUG(solver); # aug. BUG

# read reference solution
v = readdlm("PlaneSourceRaw", ',')
uEx = zeros(length(v));
for i = 1:length(v)
    if v[i] == ""
        uEx[i] = 0.0
    else
        uEx[i] = Float64(v[i])
    end
end
x = collect(range(-1.5, 1.5, length=(2 * length(v) - 1)));
uEx = [uEx[end:-1:2]; uEx]

# plot result
plt.clf()
fig = figure()
ax = gca()
lin_width = 2
ax.plot(x, uEx, "k-", linewidth=lin_width, label="exact", alpha=1.0)
ax.plot(s.xMid, uBUGc[:, 1], "b-.", linewidth=lin_width, label="cont. cons. BUG, no \$\\nabla X_0\$", alpha=0.4)
ax.plot(s.xMid, uBUG[:, 1], "g--", linewidth=lin_width, label="cont. cons. BUG, \$\\nabla X_0\$", alpha=0.4)
ax.plot(s.xMid, uBUG_cons[:, 1], "m:", linewidth=lin_width, label="cons. aug. BUG", alpha=0.9)
ax.plot(s.xMid, uBUG_non_cons[:, 1], "y-", linewidth=lin_width, label="aug. BUG", alpha=0.6)

ax.legend(loc="lower center", fontsize=13)
ax.set_xlim([s.a, s.b])
ax.tick_params("both", labelsize=14)
xlabel(L"x", fontsize=13)
ylabel(L"\Phi(x)", fontsize=13)
ax.yaxis.offsetText.set_fontsize(15)

#show()
ax.grid(true)
tight_layout();
#get x and y limits
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ratio = 0.75
#ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.savefig("scalar_flux_linesource_1d.pdf")
plt.savefig("scalar_flux_linesource_1d.png")


# plot rank
plt.clf()

fig = figure()
ax = gca()
ax.plot(rBUGc[1, :], rBUGc[2, :], "b-.", linewidth=lin_width, label="cont. cons. BUG, no \$\\nabla X_0\$", alpha=0.4)
ax.plot(rBUG[1, :], rBUG[2, :], "g--", linewidth=lin_width, label="cont. cons. BUG, \$\\nabla X_0\$", alpha=0.4)
ax.plot(rBUG_cons[1, :], rBUG_cons[2, :], "m:", linewidth=lin_width, label="cons. aug. BUG", alpha=0.9)
ax.plot(rBUG_non_cons[1, :], rBUG_non_cons[2, :], "y-", linewidth=lin_width, label="aug. BUG", alpha=0.6)

ax.legend(loc="upper left", fontsize=13)
ax.set_xlim([0, rBUG[1, end]])
ax.tick_params("both", labelsize=14)
xlabel(L"time [s]", fontsize=13)
ylabel(L"rank", fontsize=13)
ax.grid(true)
tight_layout();

#get x and y limits
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ratio = 0.75

#ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.savefig("ranks_linesource_1d.png")
plt.savefig("ranks_linesource_1d.pdf")


# plot mass
plt.clf()

font_size =20
fig = figure()
ax = gca()
ax.plot(ρBUG[1, :], abs.(ρBUGc[2, :] .- ρBUGc[2, 1]), "b-.", linewidth=lin_width, label="cont. cons. BUG, no \$\\nabla X_0\$", alpha=0.4)
ax.plot(ρ[1, :], abs.(ρ[2, :] .- ρ[2, 1]), "m:", linewidth=lin_width, label="cons. aug. BUG", alpha=0.9)
ax.legend(loc="lower left", fontsize=font_size)
ax.set_xlim([0, ρ[1, end]])
ax.set_yscale("log")

ax.tick_params("both", labelsize=font_size)
xlabel(L"t", fontsize=font_size)
ax.yaxis.offsetText.set_fontsize(font_size)
ylabel("error mass", fontsize=font_size)
ax.grid(true)
tight_layout();
plt.savefig("mass_linesource_1d.pdf")
plt.savefig("mass_linesource_1d.png")

# plot mass
plt.clf()

fig = figure()
ax = gca()
ax.plot(ρBUG[1, :], abs.(ρBUG[2, :] .- ρBUG[2, 1]), "b-.", linewidth=lin_width, label="cont. cons. BUG, \$\\nabla X_0\$", alpha=0.4)
ax.plot(ρ[1, :], abs.(ρBUGc[2, :] .- ρBUGc[2, 1]),"g--", linewidth=lin_width, label="cont. cons. BUG, no \$\\nabla X_0\$", alpha=0.4)
ax.legend(loc="center right", fontsize=font_size)
ax.set_xlim([0, ρ[1, end]])
ax.tick_params("both", labelsize=font_size+1)
xlabel(L"t", fontsize=font_size)
ax.yaxis.offsetText.set_fontsize(font_size)
ylabel("error mass", fontsize=font_size)
ax.grid(true)
ax.set_yscale("log")

tight_layout();
plt.savefig("mass_linesource_1d_no_nabla.png")
plt.savefig("mass_linesource_1d_no_nabla.pdf")

# plot mass
plt.clf()

fig = figure()
ax = gca()
ax.plot(ρ[1, :], abs.(ρ[2, :] .- ρ[2, 1]), "m:", linewidth=lin_width, label="cons. aug. BUG", alpha=0.6)
ax.plot(ρ[1, :], abs.(ρaug[2, :] .- ρaug[2, 1]), "y-", linewidth=lin_width, label="aug. BUG",alpha=0.9)
ax.legend(loc="center right", fontsize=font_size)
ax.set_xlim([0, ρ[1, end]])
ax.tick_params("both", labelsize=font_size+1)
xlabel(L"t", fontsize=font_size)
ax.yaxis.offsetText.set_fontsize(font_size)
ylabel("error mass", fontsize=font_size)
ax.grid(true)
ax.set_yscale("log")

tight_layout();
plt.savefig("mass_linesource_1d_BUGs.png")
plt.savefig("mass_linesource_1d_BUGs.pdf")

println("main finished")
