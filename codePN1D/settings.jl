__precompile__
mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    # number spatial cells
    NCells::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    # grid cell width
    Δx::Float64

    # time settings
    # end time
    tEnd::Float64;
    # time increment
    Δt::Float64;
    # CFL number 
    cfl::Float64;
    
    # degree PN
    nPN::Int64;

    # spatial grid
    x
    xMid

    # physical parameters
    σₐ::Float64;
    σₛ::Float64;

    # low rank parameters
    r::Int;

    # rank adaptivity
    epsAdapt::Float64;
    rMax::Int;

    function Settings(Nx::Int=1002,problem::String="LineSource")
        # spatial grid setting
        NCells = Nx - 1;
        a = -1.5; # left boundary
        b = 1.5; # right boundary
        
        # time settings
        tEnd = 1.0;
        cfl = 1.0; # CFL condition
        
        # number PN moments
        nPN = 500;

        x = collect(range(a,stop = b,length = NCells));
        Δx = x[2]-x[1];
        x = [x[1]-Δx;x]; # add ghost cells so that boundary cell centers lie on a and b
        x = x.+Δx/2;
        xMid = x[1:(end-1)].+0.5*Δx

        Δt = cfl*Δx;

        # physical parameters
        σₛ = 1.0;
        σₐ = 0.0;   

        r = 20;

        # parameters rank adaptivity
        epsAdapt = 0.01;
        rMax = nPN;

        # build class
        new(Nx,NCells,a,b,Δx,tEnd,Δt,cfl,nPN,x,xMid,σₐ,σₛ,r,epsAdapt,rMax);
    end

end

function IC(obj::Settings,x,xi=0.0)
    y = zeros(size(x));
    x0 = 0.0
    s1 = 0.03
    s2 = s1^2
    floor = 1e-4
    x0 = 0.0
    for j = 1:length(y);
        y[j] = max(floor,1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-x0)*(x[j]-x0))/2.0/s2))
    end
    return y;
end