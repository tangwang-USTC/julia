
"""
  Normalized moments such as n̂[nₐ], û[vth], T̂[Tₐ] and K̂[nₐTₐ]

    n̂a, T̂a, ûa = momentsGaussn(nspices,n0,nvgauss,v,w1,fvL)

  Inputs:
    nvgauss:
    v: GaussQuadrature collections
    δₜfvL: The rate of change of harmonics of distribution functions without coefficient Cf = n0/π^1.5 / vₜₕ³ as:
        δₜfₗₘ(v̂) = (2ℓ+1)//2 * Kₗ

  Outputs:
    The k-step relative rate of change of moments which are normalized by the k-step moments :
    n̂a:
    T̂a:
    ûa:

"""

function momentsGaussn(nspices::Int,n0::AbstractVector,nvgauss::AbstractVector,
                      v::AbstractVector,w1::AbstractVector,fvL::AbstractArray{T,N}) where{T,N}
    n̂a = zeros(nspices)
    ûa = zeros(nspices)
    T̂a = zeros(nspices)
    # K̂a1 = zeros(nspices)
    wlg = exp.(v)
    for isp = 1:nspices
        L1isp = 1:LM[isp] + 1
        f0 = v.^2 .* [fvL[i,1,isp] for i in nvgauss]
        f1 = v.^3 .* [fvL[i,1,isp] for i in nvgauss]
        #  # n̂a = ∫(4πv²f₀⁰(v))dv = ∫(v̂²f₀⁰(v̂))dv̂ * (4/√π nₐ)
        n̂a[isp] = w1' * (f0  .* wlg)  *(4/√π)     # [n0[isp]]

        #  # ua = 1/(3na) ∫(4πv³fₗ(v))dv = ∫(v̂³fₗ(v̂))dv̂ * (4/√π * vₜₕ/3)
        ûa[isp]  = w1' * (f1  .* wlg)  * (4//3 /√π)   # [vth]

        #  # T̂a = mₐ/2 * ∫(4πv²f₀⁰(v))dv = ∫(v̂⁴f₀⁰(v̂))dv̂ * (4/√π nₐTₐ)
        T̂a[isp]  = 2//3 * (w1' * (f0 .* v.^2  .* wlg)  *(4 /√π) - ûa[isp]^2)  # [T0[isp]]
    end
    return n̂a, ûa, T̂a
end

        # #  # Ka = mₐ/2 * ∫(4πv²f₀⁰(v))dv = ∫(v̂⁴f₀⁰(v̂))dv̂ * (4/√π nₐTₐ)
        # K̂a1[isp]  = w1' * (f0 .* v.^2  .* wlg)  *(4/√π)  # [n0[isp] * T0[isp]]

"""
  Normalized moments such as dn̂/dt[nₐ], dû/dt[vₜₕ], dTa/dt[Tk]

        dTa = momentsGaussnd(dt,nspices,û,Ta,nvgauss,v,w1,dfL)

  Inputs:
    nvgauss:
    v: GaussQuadrature collections
    T̂a: [Tk]
    δₜfvL: The rate of change of harmonics of distribution functions normalized by Cf = n0/π^1.5 / vₜₕ³ as:
        δₜfvL(v̂,ℓ) = δₜfₗₘ(v̂) = (2ℓ+1)//2 * Kₗ    (without cf)

  Outputs:
    The k-step relative rate of change of moments which are normalized by the k-step moments :
    dn̂a/dt:
    dûa/dt:
    dTa/dt:

"""

function momentsGaussnd(dt::Real,nspices::Int,û::AbstractVector,Ta::AbstractVector,nvgauss::AbstractVector,
                      v::AbstractVector,w1::AbstractVector,dfL::AbstractArray{T,N}) where{T,N}
    # dn̂a = zeros(nspices)
    dûa = zeros(nspices)
    dTa = zeros(nspices)
    wlg = exp.(v)
    for isp = 1:nspices
        df0 = v.^2 .* [dfL[i,1,isp] for i in nvgauss]
        # df1 = v.^3 .* [dfL[i,1,isp] for i in nvgauss]
        # #  # dna/dt = ∫(4πv²δₜf₀⁰(v))dv = ∫(v̂²δₜf₀⁰(v̂))dv̂ * (4/√π nₐ)
        # dn̂a[isp] = w1' * (df0  .* wlg)  *(4/√π)     # [n0[isp]]

        # #  # dûa/dt = ∫(v̂³δₜf₁(v̂))dv̂ * (4/√π /3)
        dûa[isp] = w1' * (df1  .* wlg)  * (4//3 /√π)   # vth[isp]

        # #  # T̂a = ∫(v̂⁴f₀⁰(v̂))dv̂ * (4/√π)
        dT̂a  = (w1' * (df0 .* v.^2  .* wlg))

        # dT̂a  = 2//3 *(4/√π) * (w1' * (df0 .* v.^2  .* wlg))  # [T0[isp]]
        # dTa[isp] = Ta[isp] * (dT̂a * (1 - dt * dT̂a))
        dTa[isp] = T0[isp] * 2/3 * 4/√π * dT̂a     # matlab
        # dKa[isp] = K0[isp] * (4/√π * dT̂a /(3/2 + û[isp]^2))
    end
    return dTa
end



# Rerrn = Float64.(n̂a1[isp]  - 1)
# RerrK = Float64.(K̂a1[isp]  - K̂0[isp])
# Rerru = Float64.(ûa1[isp]  - û[isp])
# RerrT = Float64.(T̂a1[isp]  - 1)

        # n̂a2 = trapezoidal1D(vremesh, vremesh.^2 .* fvL[:,1,isp]) .*(4/√π )
        # spl = Spline1D(vremesh,vremesh.^2 .* fvL[:,1,isp])
        # n̂a3 = Dierckx.integrate(spl,vremesh[1],vremesh[nv1rem]) *(4/√π )
        # Aerrn2 = Float64.(n̂a2 - 1)
        # Aerrn3 = Float64.(n̂a3 - 1)

#
# trapezoidal1D(vremesh, vremesh.^3 .* exp.(-vremesh.^2))
#
# # Factor function about û for vth of fDM
# # v̂ₜₕ = 2 × ∫₀⁹(v̂³ f₀(v̂))dv̂ = 2 I₁(f₀(v̂))= 2 × gu
# #    ≈ 2 × guTay when û ≤ 1
# ulog = -13:0.1:-2
# û = exp.(ulog)
# gu = @. 1//2 * exp(-û^2) + √π / 4 * (2û + 1 / û) * erf(û);
# guTay = @. 1 + û^2/3 - û^4/30 + û^6/210 - û^8/1512 + û^10/11880;
# Rerr = (fuTay-fu)./fu
# # plot(û,fu)
# # plot!(û,fuTay,line=(3,:dot))
# plot(û,Rerr)
