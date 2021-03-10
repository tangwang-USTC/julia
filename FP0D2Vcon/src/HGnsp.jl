"""
  H(v̂,L), G(v̂,L) = HGnspva(LM,nv1,w1,vabth,fL0,vremesh,mM,JLn1v0)
  H(v̂,u), G(v̂,L) = HGnspvau(ℓM1,LM,nv1,w1,vabth,fL0,vremesh,mM,JLn1v0,Mun)

  Calculate Rosenbluth potentials H(𝓋̂ ) and G(𝓋̂ ) through the definitions
  through Gaussian quadrature  and Newton-Cotes integrations


  Hₗₘ(vₐ) = (1+mM) * 4/√π * nb/vbth * ∑ₗ 1//(2L+1) * HL
        HL = ∑ₘ₌₋ₗᴸ Ihₗₘ Yₗₘ
  Hₗₘ(𝓋̂ ) = ∑ₗ 1//(2L+1) ∑ₘ₌₋ₗᴸ Ihₗₘ Yₗₘ
        Ihₗₘ = 1/𝓋̂ * (ILFLm + JL1FLm)
  Gₗₘ(vₐ) = 4/√π * nb*vbth * ∑ₗ 1//(2L+1) * GL
        GL = ∑ₘ₌₋ₗᴸ Igₗₘ Yₗₘ
  Gₗₘ(𝓋̂ ) = ∑ₗ 1//(2L+1) ∑ₘ₌₋ₗᴸ Igₗₘ Yₗₘ
        Igₗₘ = 1/(2L+3) * (IL2FLm + 𝓋̂ * JL1FLm) - 1/(2L-1)*(𝓋̂ * ILFLm + Jn1FLm)

  Inputs:
    LM:
    fL0: The harmonics of distribution functions normalized by (without)
        Cf = n0/π^1.5 / vₜₕ³ as:
        fL0(v̂,ℓ) = fₗₘ(v̂) = (2ℓ+1)//2 * Kₗ
    mM = mₐ/mᵦ
    vabth = vₐₜₕ/vᵦₜₕ
    nv1: nremesh

  Outputs:
    H: = Hₗₘ(𝓋̂ ) which normalized by CHL = 4/√π (1+mM)nᵦ/vᵦₜₕ
    G: = Gₗₘ(𝓋̂ ) which normalized by CGL = 4/√π nᵦvᵦₜₕ

    Question: what should be Yₗ₀ ?

"""

############################# calcluate H(v̂) and G(v̂)
function HGnspva(LM::Int64,nv1::Int64,w1::AbstractVector,vabth::Real,
         fL0::AbstractArray{T,N},vremesh::AbstractVector,mM::Real,JLn1v0::Real) where {T, N}

    LM1 = LM + 1
    H = zeros(BigFloat, nv1,LM1)
    G = zeros(BigFloat, nv1,LM1)
    # va = vremesh * vabth
    for L = 0:LM
        L1 = L + 1
        L21 = (1 // (L + L1))
        nvf90 = findlast(fL0[:,L1] .> 0 )   # verifying whether fL0 == 0
        if isnothing(nvf90) == 1
            Ih = 0
            Ig = 0
            # println("Ih = Ig=",0,",L=",L)
        else
            Ih, Ig = Ihgitp(L,nv1,w1,vabth,fL0[:,L1],vremesh,JLn1v0)
            # display(plot(va,[Ih,Ig],label=["L=",L],xscale=:log))
        end
        H[:,L1] = H[:,L1] .+ L21 * Ih
        G[:,L1] = G[:,L1] .+ L21 * Ig
    end
    return H, G
end

function HGnspvau(ℓM1::Int64,LM::Int64,nv1::Int64,w1::AbstractVector,vabth::Real,fL0::AbstractArray{T,N},
         vremesh::AbstractVector,mM::Real,JLn1v0::Real,Mun::AbstractArray{T,N},) where {T, N}

    H = zeros(BigFloat, nv1,ℓM1)
    G = zeros(BigFloat, nv1,ℓM1)
    # va = vremesh * vabth
    for L = 0:LM
        L1 = L + 1
        L21 = (1 // (L + L1))
        nvf90 = findlast(fL0[:,L1] .> 0 )   # verifying whether fL0 == 0
        if isnothing(nvf90) == 1
            Ih = 0
            Ig = 0
            # println("Ih = Ig=",0,",L=",L)
        else
            Ih, Ig = Ihgitp(L,nv1,w1,vabth,fL0[:,L1],vremesh,JLn1v0)
            # display(plot(va,[Ih,Ig],label=["L=",L],xscale=:log))
        end
        H[:,L1] = H[:,L1] .+ L21 * Ih
        G[:,L1] = G[:,L1] .+ L21 * Ig
    end
    return H * Mun , G * Mun
end

"""
  Ih, Ig = Ihgitp(L,nv1,w1,vabth,fij,vremesh,JLn1v0)

  Ihₗₘ = 1/𝓋̂ * (ILFLm + JL1FLm)
  Igₗₘ = 1/(2L+3) * (IL2FLm + 𝓋̂ * JL1FLm) - 1/(2L-1)*(𝓋̂ * ILFLm + Jn1FLm)

  Shkarofsky Integrations for Rosenbluth Potentials H(v̂) and G(v̂)
      Applying Newton-Cotes integration with interpolations for variable upper boundace integrals
      which corrected by Gaussian-Laguerre quadrature.

  Inputs:
    nv1: nremesh

  Outputs:

"""
# L = 0
function Ihgitp(L::Int64,nv1::Int64,w1::AbstractVector,vabth::Real,
                 fij::AbstractVector,vremesh::AbstractVector,JLn1v0::Real)

    kSpline1D = 2  # best for accomulation errors
    IL0v0 = √π / 4
    vn1 = vremesh.^(L-1)
    vL = vn1 .* vremesh
    vL1 = vL .* vremesh
    vL2 = vL1 .* vremesh
       va = vremesh * vabth
       vaL = va.^L
       vaL1 = vaL .* va
    # interpolations ∫(v̂ᵦʲv̂ᵦ²f(v̂ᵦ))dv̂ᵦ for variable boundace integrals with interpolations
    fij = fij .* vremesh.^2
    # #### v ∈ [0:va]  , interpolation when integration
    IL0FLm = cumILFLMlogitp(vremesh,fij .* vL ,va) ./ vaL
    if L == 0
        IL0FLm = IL0FLm * ( IL0v0/ IL0FLm[nv1])
    end
    # println("L=",L,",R=",( IL0v0/ IL0FLm[nv1]))
      # I00(v→∞) = c > 0
    IL2FLm = cumILFLMlogitp(vremesh,fij .* vL2,va) ./ vaL1
      # I02(v→∞) = 0
    # #### v ∈ [va:∞]  , integration of fij = fvu[:,L1,isp]
    JL1FLm = - reverse(cumul_integrate(reverse(vremesh),reverse(fij ./ vL1))) .* vL1
     if va[nv1] ≤ vremesh[nv1] # vath ≥ vbth
         # itpDL = QuadraticInterpolation(JL1FLm,vremesh)
         itpDL = QuadraticInterpolation([0;JL1FLm],[0;vremesh])
     else
         # itpDL = Spline1D(vremesh,JL1FLm;k=kSpline1D,bc="extrapolate")
         itpDL = Spline1D([0;vremesh],[0;JL1FLm];k=kSpline1D,bc="extrapolate")
     end
# display(plot(vremesh,JL1FLm,label= ["v,L=",L]))
     JL1FLm = itpDL.(va)
# display(plot(va,JL1FLm,xscale=:log,label= ["va,L=",L]))
    JLn1FLm =  - reverse(cumul_integrate(reverse(vremesh),reverse(fij ./ vn1))) .* vL
    # J0n1(v → 0) = c > 0
    # to interpolate J(v̂) into J(𝓋̂ )
    if L == 0
        JLn1FLm=JLn1FLm * (JLn1v0/ JLn1FLm[1])
        if va[nv1] ≤ vremesh[nv1]
            itpDL = QuadraticInterpolation([JLn1v0;JLn1FLm],[0;vremesh])
        else
            itpDL = Spline1D(vremesh,JLn1FLm;k=kSpline1D,bc="extrapolate")
        end
    else
        if va[nv1] ≤ vremesh[nv1]
            itpDL = QuadraticInterpolation([0;JLn1FLm],[0;vremesh])
            # itpDL = Spline1D([0;vremesh],[0;JLn1FLm];k=kSpline1D)
        else
            # println("va=",Float64.(va[nv1]),",v=",Float64.(vremesh[nv1]))
            itpDL = Spline1D([0;vremesh],[0;JLn1FLm];k=kSpline1D,bc="extrapolate")
        end
    end
         JLn1FLm = itpDL.(va)
    # display(plot(va,JLn1FLm,xscale=:log,label= ["va,L=",L]))
    # p0 = plot(vremesh,[IL0FLm,IL2FLm],label=["L=",L])
    # p0 = plot(vremesh,[JL1FLm,JLn1FLm],label=["L=",L],xscale=:log)
    # p0 = plot(va,[abs.(JL1FLm),abs.(JLn1FLm)/8],label=["L=",L],xscale=:log,yscale=:log)
    # p0 = plot(va,[IL0FLm,IL2FLm,JL1FLm,JLn1FLm],xscale=:log,label=["L=",L])
    # display(p0)
    #### renormal JLFLM by the vaule of Gaussian-Laguerre quadrature when v → ∞
    # flag = transpose(w1 .* (fij .* exp.(v) ./ v.^α))  # wlag = @. exp(v) / v^α
    # IL1FLmg = flag * (1 ./ vL1 ) .* vaL1  # FLm / v^(L+1) .* va^(L+1) , v ∈ [0:∞]
    # ILn1FLmg =flag * (1 ./ vn1 ) .* van1  # FLm / v^(L-1) .* va^(L-1) , v ∈ [0:∞]
    # IL0FLmg = flag * vL ./ vaL     # FLm * v^(L) ./ va^(L)     , v ∈ [0:∞]
    # IL2FLmg = flag * vL2 ./vaL2    # FLm * v^(L+2) ./ va^(L+2) , v ∈ [0:∞]
    #
    # IL2FLm = IL2FLm * (IL2FLmg[nv1] / IL2FLm[nv1])
    # JL1FLm = JL1FLm * (IL1FLmg[1] / JL1FLm[1])
    # println(Float64.([IL0FLmg[nv1] / IL0FLm[nv1],IL2FLmg[nv1] / IL2FLm[nv1],IL1FLmg[1] / JL1FLm[1],ILn1FLmg[1]/ JLn1FLm[1]]))
    # #######  Ih and Ig
    Ih = IL0FLm ./ va + JL1FLm ./ va
    # Ig = 1 // 3 * (IL2FLm + JL1FLm * va) +
    #             (JLn1FLm + IL0FLm * va)
    Ig = 1 // (2L + 3) * (IL2FLm + JL1FLm .* va) -
          1 // (2L - 1) * (JLn1FLm + IL0FLm .* va)
# p0 = plot(va,Ig,xscale=:log,label=["L=",L])
# display(p0)
    return Ih, Ig
end

"""
  ILFLMitp = cumILFLMlogitp(vremesh,dILFLm,va)
    where v ∈ [0:va] and iterpolation from v = 0 where f(v=0) = 0

  JLFLMitp = cumJLFLMlogitp(vremesh,dILFLm,va)
    where v ∈ [va:∞]

  inputs:

  outputs:

"""

# function cumJLFLMlogitp(vremesh::AbstractVector,dILFLm::AbstractVector,va::AbstractVector)
#     # JLFLMitp = ∫ᵥ^∞(dJLFLm)dv
#     FLmremesh = - reverse(cumul_integrate(reverse(vremesh),reverse(FLmremesh)))
#     return FLmremesh
# end

function cumILFLMlogitp(vremesh::AbstractVector,dILFLm::AbstractVector,va::AbstractVector)

    kSpline1D = 3
    # ILFLMitp = ∫₀ᵛ(dILFLm)dv
    dILFLm = cumul_integrate(vremesh,dILFLm)
    # # interpolate to va
    if va[end] ≤ vremesh[end]
        # println(Float64.(vremesh[1:4]))
        # itpDL = QuadraticInterpolation(dILFLm,vremesh)
        itpDL = QuadraticInterpolation([0.0;dILFLm],[0.0;vremesh])
        # itpDL = Spline1D([0;vremesh],[0;dILFLm];k=kSpline1D)
        # p0 = plot(vremesh,dILFLm)
        # p1 = plot(va,itpDL.(va))
        # display(plot(p0,p1,layout =(2,1)))
        return itpDL.(va)
    else
        itpDL = Spline1D([0.0;vremesh],[0.0;dILFLm];k=kSpline1D,bc="extrapolate")
        # p0 = plot(vremesh,dILFLm)
        # p1 = plot(va,itpDL(va))
        # display(plot(p0,p1,layout =(2,1)))
        return itpDL(va)
    end
end
