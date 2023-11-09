__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using PyCall

struct Solver
    # spatial grid of cell interfaces
    x::Array{Float64};

    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    γ::Array{Float64,1};
    # flux matrix PN system
    A::Array{Float64,2};
    # Roe matrix
    AbsA::Array{Float64,2};

    # stencil matrices
    Dₓ::Tridiagonal{Float64, Vector{Float64}};
    Dₓₓ::Tridiagonal{Float64, Vector{Float64}};

    # physical parameters
    σₐ::Float64;
    σₛ::Float64;

    # constructor
    function Solver(settings)
        x = settings.x;
        nx = settings.NCells;
        Δx = settings.Δx;

        # setup flux matrix
        γ = ones(settings.nPN);

        # setup γ vector
        γ = zeros(settings.nPN);
        for i = 1:settings.nPN
            n = i-1;
            γ[i] = 2/(2*n+1);
        end

        # setup PN system matrix
        upper = [(n+1)/(2*n+1)*sqrt(γ[n+2])/sqrt(γ[n+1]) for n = 0:settings.nPN-2];
        lower = [n/(2*n+1)*sqrt(γ[n])/sqrt(γ[n+1]) for n = 1:settings.nPN-1];
        A = Tridiagonal(lower,zeros(settings.nPN),upper)
        
        # setup Roe matrix
        S = eigvals(Matrix(A))
        V = eigvecs(Matrix(A))
        AbsA = V*abs.(Diagonal(S))*inv(V)

        # set up spatial stencil matrices
        Dₓ = Tridiagonal(-[ones(nx-2); 0]./Δx/2.0,zeros(nx),[0; ones(nx-2)]./Δx/2.0) # central difference matrix
        Dₓₓ = Tridiagonal([ones(nx-2);0]./Δx/2.0,-[0; ones(nx-2); 0]./Δx,[0; ones(nx-2)]./Δx/2.0) # stabilization matrix

        new(x,settings,γ,A,AbsA,Dₓ,Dₓₓ,settings.σₐ,settings.σₛ);
    end
end

py"""
import numpy
def qr(A):
    return numpy.linalg.qr(A)
"""

function SetupIC(obj::Solver)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1] = 2.0/sqrt(obj.γ[1])*IC(obj.settings,obj.settings.xMid);
    return u;
end

# full PN method
function Solve(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN;
    nx = obj.settings.NCells;

    # Set up initial condition
    u = SetupIC(obj);

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;
    A = obj.A;
    AbsA = obj.AbsA;

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        u = u .- Δt * obj.Dₓ*u*A' .+ Δt * obj.Dₓₓ*u*AbsA' .- Δt * σₐ*u .- Δt * σₛ*u*G; 
        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*u;

end

# projector splitting integrator (stabilized)
function SolvePSI(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        # K-step
        K = X*S;
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;
        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,tildeS1,tildeS2 = svd!(K); tildeS = Diagonal(tildeS1)*tildeS2'; # use svd instead of QR since svd is more efficient in Julia

        # S-step
        XDₓₓX = X1'*obj.Dₓₓ*X1
        XDₓX = X1'*obj.Dₓ*X1
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        S = tildeS .+ Δt * XDₓX*tildeS*WAW .+ Δt * XDₓₓX*tildeS*WAbsAW .+ Δt * XσₐX*tildeS .+ Δt * XσₛX*tildeS*WGW;

        # L-step
        L = W*S';
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; 
        W1,S2,S1 = svd!(L); S .= (Diagonal(S2)*S1')'; # use svd instead of QR since svd is more efficient in Julia

        X .= X1; W .= W1;
        
        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W';

end

# unconventional integrator
function SolveUI(obj::Solver)
    # store values from settings into function
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r; # rank

    # time settings
    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    # spatial and moment settings
    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    # compute scattering interaction cross-sections
    G = Diagonal([0.0;ones(N-1)]); # scattering diagonal
    σₛ = Diagonal(ones(nx)).*obj.settings.σₛ;   # scattering
    σₐ = Diagonal(ones(nx)).*obj.settings.σₐ;   # absorption

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        # K-step
        K = X*S; 
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;

        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,_,_ = svd(K); # use svd instead of QR since svd is more efficient in Julia
        M = X1'*X;

        # L-step
        L = W*S';
        XDₓₓX = X'*obj.Dₓₓ*X;
        XDₓX = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L
        W1,_,_ = svd(L); # use svd instead of QR since svd is more efficient in Julia
        N = W1'*W;

        # S-step
        S = M*S*N';
        XDₓₓX = X1'*obj.Dₓₓ*X1;
        XDₓX = X1'*obj.Dₓ*X1;
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        WAW = W1'*A'*W1;
        WAbsAW = W1'*AbsA'*W1;
        WGW = W1'*G*W1;

        S = S .- Δt * XDₓX*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW; # advance S

        X .= X1; W .= W1;
        
        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W';

end

# rank adaptive unconventional integrator
function SolveBUG_cons_Lukas(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    # ensure that e1 is first column in W matrix
    e1 = [1.0;zeros(obj.settings.nPN-1)];
    W1,_ = py"qr"([e1 W])
    S = S*(W'*W1);
    W = W1;

    #Compute diagonal of scattering matrix G
    G =Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    ρ = zeros(2,nt);
    rankInTime = zeros(2,nt);

    # number of conserved quantities
    m = 1;

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        ρ[1,n] = n * Δt;
        rankInTime[1,n] = n * Δt;
        rankInTime[2,n] = r;
        for j = 1:size(X,1)
            ρ[2,n] += X[j,:]'*S*W[1,:];
        end
    
        # K-step
        K = X*S;
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;
        advectionTerm = - obj.Dₓ*K*WAW .+ obj.Dₓₓ*K*WAbsAW
        K = K .+ Δt * advectionTerm .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,_ = py"qr"([K X advectionTerm]); # use svd instead of QR since svd is more efficient in Julia
        M = X1'*X;

        # L-step
        St = S[:, (m+1):r]
        W_cons = W[:, (m+1):r]
        T = St'*St
        L = W*S';
        L_cons = W_cons*T

        XDₓₓX = X'*obj.Dₓₓ*X;
        X₁ᵀDₓX₁ = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        L_cons .= L_cons .- Δt * A*L*X₁ᵀDₓX₁'*St .+ Δt * AbsA*L*XDₓₓX'*St .- Δt * L*XσₐX*St .- Δt *G'*L*XσₛX*St; # advance L
        L_cons .= L_cons .+ Δt * W*WAW'*S'*X₁ᵀDₓX₁*St .- Δt * W*WAbsAW'*S'*XDₓₓX*St .+ Δt * W*S'*XσₐX'*St .+ Δt * W*WGW'*S'*XσₛX'*St;# advance L
        W1,_ = py"qr"([W L_cons]); 
        W1,_ = py"qr"([e1 W1]);
        N = W1'*W;

        # S-step
        S = M*S*N';
        XDₓₓX = X1'*obj.Dₓₓ*X1;
        X₁ᵀDₓX₁ = X1'*obj.Dₓ*X1;
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        WAW = W1'*A'*W1;
        WAbsAW = W1'*AbsA'*W1;
        WGW = W1'*G*W1;

        S .= S .- Δt * X₁ᵀDₓX₁*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW;# advance S
    
        # truncate
        X, S, W, r = truncateConservative!(obj,X1,S,W1);

        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W',ρ,rankInTime;

end

# rank adaptive unconventional integrator
function SolveBUG_cont_cons(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    # ensure that e1 is first column in W matrix
    e1 = [1.0;zeros(obj.settings.nPN-1)];
    W1,_ = py"qr"([e1 W])
    S = S*(W'*W1);
    W = W1;

    #Compute diagonal of scattering matrix G
    G =Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    ρ = zeros(2,nt);
    rankInTime = zeros(2,nt);

    # number of conserved quantities
    m = 1;

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        ρ[1,n] = n * Δt;
        rankInTime[1,n] = n * Δt;
        rankInTime[2,n] = r;
        for j = 1:size(X,1)
            ρ[2,n] += X[j,:]'*S*W[1,:];
        end
    
        # K-step
        K = X*S;
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;
        advectionTerm = - obj.Dₓ*K*WAW .+ obj.Dₓₓ*K*WAbsAW
        K = K .+ Δt * advectionTerm .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,_ = py"qr"([K X]); # use svd instead of QR since svd is more efficient in Julia
        M = X1'*X;

        # L-step
        St = S[:, (m+1):r]
        W_cons = W[:, (m+1):r]
        T = St'*St
        L = W*S';
        L_cons = W_cons*T

        XDₓₓX = X'*obj.Dₓₓ*X;
        X₁ᵀDₓX₁ = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        L_cons .= L_cons .- Δt * A*L*X₁ᵀDₓX₁'*St .+ Δt * AbsA*L*XDₓₓX'*St .- Δt * L*XσₐX*St .- Δt *G'*L*XσₛX*St; # advance L
        L_cons .= L_cons .+ Δt * W*WAW'*S'*X₁ᵀDₓX₁*St .- Δt * W*WAbsAW'*S'*XDₓₓX*St .+ Δt * W*S'*XσₐX'*St .+ Δt * W*WGW'*S'*XσₛX'*St;# advance L
        W1,_ = py"qr"([W L_cons]); 
        W1,_ = py"qr"([e1 W1]);
        N = W1'*W;

        # S-step
        S = M*S*N';
        XDₓₓX = X1'*obj.Dₓₓ*X1;
        X₁ᵀDₓX₁ = X1'*obj.Dₓ*X1;
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        WAW = W1'*A'*W1;
        WAbsAW = W1'*AbsA'*W1;
        WGW = W1'*G*W1;

        S .= S .- Δt * X₁ᵀDₓX₁*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW;# advance S
    
        # truncate
        X, S, W, r = truncateConservative!(obj,X1,S,W1);

        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W',ρ,rankInTime;

end


# rank adaptive unconventional integrator
function SolveBUG_cons(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    # ensure that e1 is first column in W matrix
    e1 = [1.0;zeros(obj.settings.nPN-1)];
    W1,_ = py"qr"([e1 W])
    S = S*(W'*W1);
    W = W1;

    #Compute diagonal of scattering matrix G
    G =Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    ρ = zeros(2,nt);
    rankInTime = zeros(2,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        ρ[1,n] = n * Δt;
        rankInTime[1,n] = n * Δt;
        rankInTime[2,n] = r;
        for j = 1:size(X,1)
            ρ[2,n] += X[j,:]'*S*W[1,:];
        end

        # K-step
        K = X*S;
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;
        advectionTerm = - obj.Dₓ*K*WAW .+ obj.Dₓₓ*K*WAbsAW
        K = K .+ Δt * advectionTerm .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,_ = py"qr"([X K]); # use svd instead of QR since svd is more efficient in Julia
        M = X1'*X;

        # L-step
        XDₓₓX = X'*obj.Dₓₓ*X;
        XDₓX = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        L = W*S';
        L .= L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L
        W1,_ = py"qr"([W L]); # use svd instead of QR since svd is more efficient in Julia
        N = W1'*W;

        # S-step
        S = M*S*N';
        XDₓₓX = X1'*obj.Dₓₓ*X1;
        X₁ᵀDₓX₁ = X1'*obj.Dₓ*X1;
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        WAW = W1'*A'*W1;
        WAbsAW = W1'*AbsA'*W1;
        WGW = W1'*G*W1;

        S .= S .- Δt * X₁ᵀDₓX₁*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW;# advance S
    
        # truncate
        X, S, W, r = truncateConservative!(obj,X1,S,W1);

        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W',ρ,rankInTime;

end

# rank adaptive unconventional integrator
function SolveBUG(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    ρ = zeros(2, nt);
    rankInTime = zeros(2,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        ρ[1,n] = n * Δt;
        for j = 1:size(X,1)
            ρ[2,n] += X[j,:]'*S*W[1,:];
        end
        
        rankInTime[1,n] = n * Δt;
        rankInTime[2,n] = r;
        # K-step
        K = X*S;
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;
        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,_,_ = svd([K X]); # use svd instead of QR since svd is more efficient in Julia
        M = X1'*X;

        # L-step
        XDₓₓX = X'*obj.Dₓₓ*X;
        XDₓX = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        L = W*S';
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L
        W1,_,_ = svd([L W]); # use svd instead of QR since svd is more efficient in Julia
        N = W1'*W;

        # S-step
        S = M*S*N';
        XDₓₓX = X1'*obj.Dₓₓ*X1;
        XDₓX = X1'*obj.Dₓ*X1;
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        WAW = W1'*A'*W1;
        WAbsAW = W1'*AbsA'*W1;
        WGW = W1'*G*W1;
        S = S .- Δt * XDₓX*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW; # advance S

        # truncate
        X, S, W, r = truncate!(obj,X1,S,W1);

        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W',ρ, rankInTime;

end

# new parallel integrator
function SolveParallel(obj::Solver)
    # store values from settings into function
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r; # rank

    # time settings
    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    # spatial and moment settings
    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    # compute scattering interaction cross-sections
    G = Diagonal([0.0;ones(N-1)]); # scattering diagonal
    σₛ = Diagonal(ones(nx)).*obj.settings.σₛ;   # scattering
    σₐ = Diagonal(ones(nx)).*obj.settings.σₐ;   # absorption

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    rVec = zeros(2,nt)

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        r = size(S,1);

        # K-step (parallel)
        K = X*S; 
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;

        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K

        Xtmp,_ = py"qr"([X K]); X1Tilde = Xtmp[:,(r+1):end];

        # L-step (parallel)
        L = W*S';
        XDₓₓX = X'*obj.Dₓₓ*X;
        XDₓX = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L

        Wtmp,_ = py"qr"([W L]); W1Tilde = Wtmp[:,(r+1):end];

        # S-step (parallel)
        S = S .- Δt * XDₓX*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW; # advance S

        SNew = zeros(2 * r, 2 * r);

        SNew[1:r,1:r] = S;
        SNew[(r+1):end,1:r] = X1Tilde'*K;
        SNew[1:r,(r+1):end] = L' * W1Tilde;

        # truncate
        X, S, W = truncate!(obj,[X X1Tilde],SNew,[W W1Tilde]);
        rVec[1,n] = t;
        rVec[2,n] = r;

        t += Δt;

        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W', rVec;

end

function truncate!(obj::Solver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(S);
    rmax = -1;
    rMaxTotal = obj.settings.rMax;
    S .= zeros(size(S));

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(D);

    rmax = Int(floor(size(D,1)/2));

    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if tmp < tol
            rmax = j;
            break;
        end
    end

    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,2);

    for l = 1:rmax
        S[l,l] = D[l];
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    # update solution with new rank
    X = X*U;
    W = W*V;

    # return rank
    return X[:,1:rmax],S[1:rmax,1:rmax],W[:,1:rmax],rmax;
end

function truncateConservative!(obj::Solver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    r0 = size(S,1);
    rMaxTotal = obj.settings.rMax;

    # ensure that e1 is first column in W matrix, most likely not needed since conservative basis is preserved. Ensure cons basis is in front
    e1 = [1.0;zeros(obj.settings.nPN-1)];
    W1,_ = py"qr"([e1 W])
    S = S*(W'*W1);
    W = W1;
    K = X*S;

    # split solution in conservative and remainder
    Kcons = K[:,1]; Krem = K[:,2:end];
    Wcons = W[:,1]; Wrem = W[:,2:end];
    Xcons = Kcons ./ norm(Kcons); Scons = norm(Kcons);
    Xrem,Srem = py"qr"(Krem);

    # truncate remainder part and leave conservative part as is
    U,Sigma,V = svd(Srem);
    rmax = -1;

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(Sigma);

    rmax = Int(floor(size(Sigma,1)/2));

    for j=1:2*rmax
        tmp = sqrt(sum(Sigma[j:2*rmax]).^2);
        if(tmp<tol)
            rmax = j;
            break;
        end
    end

    rmax = min(rmax,rMaxTotal);
    r1 = max(rmax,2);

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    Srem = Diagonal(Sigma[1:r1]);
    Xrem = Xrem * U[:,1:r1];
    Wrem = Wrem * V[:,1:r1];
    What = [e1 Wrem];
    Xhat = [Xcons Xrem];
    Xnew,R1 = py"qr"(Xhat);
    Wnew,R2 = py"qr"(What);
    Snew = R1*[Scons zeros(1,r1); zeros(r1,1) Srem]*R2';
    return Xnew, Snew, Wnew, r1;
end
