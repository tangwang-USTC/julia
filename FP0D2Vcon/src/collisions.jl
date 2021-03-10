
"""
  dft = collisionL0(nspices,nSf,nSdft,nv1rem,ℓM1,LM,vth,m0,Zq,n0,vremesh,μ,DP,fvu,Hvu,Gvu,Mμ)
  dftc = collisionL0c(nspices,nSf,nSdft,nv1rem,ℓM1,LM,vth,m0,Zq,n0,vremesh,μ,DP,fvu,Hvu,Gvu,Mμ)

  Γₐ =

  Inputs:
    v: vremesh
    m0: non-normalization
    fvu(v̂,μ)：The normalized distribution function
              without Cf = n0/π^1.5 / vₜₕ³ due to fvu(v̂,μ) = fvL(v̂,μ) * Mun

  Output:
    δfₗₘ(v)/δt = δfₗₘ(v̂)/δt, without (or with) coefficients cf

"""

function collisionL0c(nspices::Int,nSf::Int,nSdft::Int,nv1::Int,ℓM1::Int,LM::AbstractVector,
    vth::AbstractVector,m0::AbstractVector,Zq::AbstractVector,n0::AbstractVector,v::AbstractVector,
    μ::AbstractVector,DP::AbstractArray{T4,N},fvu::AbstractArray{T3,N},Hvu::AbstractArray{T3,N},
    Gvu::AbstractArray{T3,N},Mμ::AbstractArray{T2,N2}) where {T,T2,T3,T4,N,N2}

    mu = transpose(μ)
    sin2 = 1 .- mu.^2
    v2 = v.^2
    ######################## δft = Γa ∑ᵢSᵢ,  only the summation output
    if nSdft == 1
        δft = zeros(BigFloat,nv1,ℓM1,nspices)
        cs = ones(6)
        #### δfₗ/δt = δft * Mμ
        for isp in 1:nspices
            vath = vth[isp]
            nsp_vec = 1:nspices
            ############ SF = Mvn * XLm * Mun , X = F, H ,G, X = X(v̂)
            nspF = nsp_vec[nsp_vec .≠ isp]
            # if length(nspF) == 1
                iFv = nspF[1]
                mM = m0[isp] / m0[iFv]
                nb = n0[iFv]
                vbth = vth[iFv]
                vabth = vath / vbth
                qma = qma0 * Zq[isp] * Zq[iFv] / (m0[isp]/ Da)
                lnAab = lnA()
                Γa = 4π * qma^2 * lnAab
            #### coefficients
            CfL = n0[isp] / π^1.5 / vath^3  # Normalized coefficients of fₗₘ
            Γa = Γa * CfL
            CFL = nb / π^1.5 / vbth^3  # Normalized coefficients of Fₗₘ
            CHL = nb * (1+mM) * 4/√π /vbth  # Normalized coefficients of Hₗₘ
            CGL = nb * 4/√π * vbth           # Normalized coefficients of Gₗₘ
            CfFL = Γa * CFL * 4π * mM
            CfHL = Γa * CHL * (1 - mM) / (1 + mM)
            CfGL = Γa * CGL * (1 // 2)
            va3vb1 = vath^3 * vbth
            ###
            cs[1] = CfFL
            cs[2] = CfHL / (vath * vbth)
            cs[3] = CfHL /vath^2         #     ./ v2
            cs[4] = CfGL / (vath * vbth)^2
            cs[5] = CfGL /vath^4         #    ./ v2.^2
            cs[6] =  CfGL/va3vb1     # = cs[7] = cs[8]     ./ v2
            #######
            Sf, SF = SfFL0(nspices,isp,nSf,nv1,ℓM1,LM,vth,m0,v,μ,DP,fvu,Hvu,Gvu)  # vabth
            for i in [1,2,4]
                δft[:,:,isp] += cs[i] * (Sf[:,:,i] .* SF[:,:,i])
            end
            i = 3
            δft[:,:,isp] += cs[i] ./ v2 .* (Sf[:,:,i] .* SF[:,:,i])
            i = 5   #  ./ v2.^2 however the numerical errors blows up the data when v̂ ≪ 1
                    #  ./v.^(2~3) maybe is a resonable which is right qualitatively without the absolute value.
                    #  δft[0,:,0,isp] ∝ 1/v, affect df/dt when Ta ~ Tb and ua ~ ub
            δft[:,:,isp] += cs[5]  ./ v2.^2 .* (Sf[:,:,i] .* SF[:,:,i])
            for i in [6,7,8]
                δft[:,:,isp] += cs[6] ./ v2 .* (Sf[:,:,i] .* SF[:,:,i])
            end
            ## Project δft(v̂,μ,isp) to δfₗ/δt(v̂,ℓ,isp)
            δft[:,:,isp] = δft[:,:,isp] * Mμ  # δfₗ/δt
        end  # for isp
        return δft
    else  ######################### δft(8)
        δft = zeros(BigFloat,nv1,ℓM1,nSf,nspices)
        cs = ones(6)
        #### δfₗ/δt = δft * Mμ
        for isp in 1:nspices
            vath = vth[isp]
            nsp_vec = 1:nspices
            ############ SF = Mvn * XLm * Mun , X = F, H ,G, X = X(v̂)
            nspF = nsp_vec[nsp_vec .≠ isp]
            # if length(nspF) == 1
                iFv = nspF[1]
                mM = m0[isp] / m0[iFv]
                nb = n0[iFv]
                vbth = vth[iFv]
                vabth = vath / vbth

                qma = qma0 * Zq[isp] * Zq[iFv] / (m0[isp]/ Da)
                lnAab = lnA()
                Γa = 4π * qma^2 * lnAab
            #### coefficients
            CfL = n0[isp] / π^1.5 / vath^3  # Normalized coefficients of fₗₘ
            Γa = Γa * CfL
            CFL = nb / π^1.5 / vbth^3  # Normalized coefficients of Fₗₘ
            CHL = nb * (1+mM) * 4/√π /vbth  # Normalized coefficients of Hₗₘ
            CGL = nb * 4/√π * vbth           # Normalized coefficients of Gₗₘ
            CfFL = Γa * CFL * 4π * mM
            CfHL = Γa * CHL * (1 - mM) / (1 + mM)
            CfGL = Γa * CGL * (1 // 2)
            va3vb1 = vath^3 * vbth
            ###
            cs[1] = CfFL
            cs[2] = CfHL / (vath * vbth)
            cs[3] = CfHL /vath^2         #     ./ v2
            cs[4] = CfGL / (vath * vbth)^2
            cs[5] = CfGL /vath^4         #    ./ v2.^2
            cs[6] =  CfGL/va3vb1     # = cs[7] = cs[8]     ./ v2
            Sf, SF = SfFL0(nspices,isp,nSf,nv1,ℓM1,LM,vth,m0,v,μ,DP,fvu,Hvu,Gvu)
            for i in [1,2,4]
                δft[:,:,i,isp] = cs[i] * (Sf[:,:,i] .* SF[:,:,i]) * Mμ
            end
            i = 3
            δft[:,:,i,isp] = cs[i] ./ v2 .* (Sf[:,:,i] .* SF[:,:,i]) * Mμ
            i = 5   #  ./ v2.^2 however the numerical errors blows up the data when v̂ ≪ 1
                    #  ./v.^(2~3) maybe is a resonable which is right qualitatively without the absolute value.
                    #  δft[v→0,:,0,isp] ∝ 1/v^3 affect df/dt when Ta ~ Tb and ua ~ ub
            δft[:,:,i,isp] = cs[5]  ./ v2.^2 .* (Sf[:,:,i] .* SF[:,:,i]) * Mμ
            for i in [6,7,8]
                δft[:,:,i,isp] = cs[6] ./ v2 .* (Sf[:,:,i] .* SF[:,:,i]) * Mμ
            end
            vlog = log.(v)
            p1 = plot(vlog,δft[:,:,1,isp],label=["isf=",1])
            p2 = plot(vlog,δft[:,:,2,isp],label=["isf=",2])
            # p3 = plot(vlog,δft[:,:,3,isp]*n20,label=["isf=",3])
            p4 = plot(vlog,δft[:,:,4,isp],label=["isf=",4])
            # p5 = plot(vlog,δft[:,:,5,isp]*n20,label=["isf=",5])
            p6 = plot(vlog,δft[:,:,6,isp]*n20,label="isf=6")
            p7 = plot(vlog,δft[:,:,7,isp],label="isf=7")
            # p8 = plot(vlog,δft[:,1,8,isp],label=["isf=",8])
            # # δft = ∑ᵢSᵢ
            for i = 2:nSf # [2,3,4,6,7,8]
                δft[:,:,1,isp] = δft[:,:,1,isp] + δft[:,:,i,isp]
            end
            ps = plot(vlog,δft[:,:,1,isp]-δft[:,:,5,isp],label=["isf=",9])
            display(plot(p1,p2,p4,p6,p7,ps,layout=(3,2),legend=false))
        end  # for isp
        return δft[:,:,1,:]
    end # nSdft == 1
end

# Sf[abs.(Sf) .< 1e-11] .= 0
# SF[abs.(SF) .< 1e-11] .= 0
# pf3 = plot(v,Sf[:,:,3],xscale=:log,legend=false)
# pf4 = plot(v,Sf[:,:,4],xscale=:log,legend=false)
# pf5 = plot(v,Sf[:,:,5],xscale=:log,label="Sf=5",legend=:topleft)
# pf6 = plot(v,Sf[:,:,6],xscale=:log,legend=false)
# pF3 = plot(v,SF[:,:,3],xscale=:log,legend=false)
# pF4 = plot(v,SF[:,:,4],xscale=:log,legend=false)
# pF5 = plot(v,SF[:,:,5],xscale=:log,legend=false)
# pF6 = plot(v,SF[:,:,6],xscale=:log,legend=false)
# # display(plot(pf3,pf4,pf5,pf6,pF3,pF4,pF5,pF6,layout=(2,4)))
# display(plot(pf3,pF3,pf4,pF4,pf5,pF5,pf6,pF6,layout=(4,2)))
# # display(plot(pf4,pF4,layout=(2,1)))

"""
  Sf, SF = SfFL0(nspices,isp,nSf,nv1,ℓM1,LM,vth,m0,v,μ,DP,fvu,Hvu,Gvu)

  Or
  δft[:,:,nSf] = SfFL0(nspices,isp,nSf,nv1,ℓM1,LM,vth,m0,v,μ,DP,fvu,Hvu,Gvu)
  where δft = Sf .* SF.

  Inputs:
    v: vremesh
"""

# non-linerized model for axisymmetric distribution fL0[nv,nL,nspices]
# m == 0
function SfFL0(nspices::Int,isp::Int,nSf::Int,nv1::Int,ℓM1::Int,LM::AbstractVector,vth::AbstractVector,
    m0::AbstractVector,v::AbstractVector,μ::AbstractVector,DP::AbstractArray{T4,N},
    fvu::AbstractArray{T3,N},Hvu::AbstractArray{T3,N},Gvu::AbstractArray{T3,N}) where {T,T3,T4,N}

    Sf = zeros(BigFloat,nv1,ℓM1,nSf)
    SF = zeros(BigFloat,nv1,ℓM1,nSf)
    mu = transpose(μ)
    sin2 = (1 .-mu.^2)
    vath = vth[isp]
    nsp_vec = 1:nspices
    ############ SF = Mvn * XLm * Mun , X = F, H ,G, X = X(v̂)
    nspF = nsp_vec[nsp_vec .≠ isp]
    if length(nspF) == 1
        iFv = nspF[1]
        mM = m0[isp] / m0[iFv]
        vbth = vth[iFv]
        vabth = vath / vbth
        dva = abs(vabth - 1)  # abs(va[nv1] - v[nv1]) / v[nv1]
        va = v * vabth        # 𝓋̂ = v/vbth = (v̂ * vath) / vbth
        #### for the derivatives of dⁿH/dvⁿ and dⁿG/dvⁿ, H = H(𝓋̂ )
        ##### SF2 = D^(1) * Hvu[:,:,isp] , inapplicable for H and dH = zeros(nv1,ℓM1)
        df = zeros(nv1,ℓM1)
        ddf = zeros(nv1,ℓM1)
        dH = zeros(nv1,ℓM1)  #
        dG = zeros(nv1,ℓM1)
        ddG = zeros(nv1,ℓM1)
        fvu_c = 1e-4         # Critical vaule of f(v̂→0,u) for df/dv due to boundary error of f(v,ℓ) * Mμ
        for iu in 1:ℓM1
            spl = Spline1D(v,fvu[:,iu,isp])
            df[:,iu] = derivative(spl,v)
            if fvu[4,iu,isp] < fvu_c
                df[1:3,iu] .= df[4,iu]
            end
            spl = Spline1D(v,df[:,iu])
            ddf[:,iu] = derivative(spl,v)
        end
        for iu in 1:ℓM1
            spl = Spline1D(va,Hvu[:,iu,isp])
            dH[:,iu] = derivative(spl,va)
            spl = Spline1D(va,Gvu[:,iu,isp])
            dG[:,iu] = derivative(spl,va)
            # λ = 0.001
            # spl = fit(SmoothingSpline, Float64.(v), Float64.(dG[:,iu]), λ) # bigger λ means the less freedom
            # dG[:,iu] = SmoothingSplines.predict(spl) # fitted vector
            spl = Spline1D(va,dG[:,iu])
            ddG[:,iu] = derivative(spl,va;nu=1)
            if va[nv1] ≤ v[nv1]  # vath ≥ vbth
                itpDL = QuadraticInterpolation(fvu[:,iu,iFv],v)
                SF[:,iu,1] = itpDL.(va)
            else
                itpDL = Spline1D(v,fvu[:,iu,iFv];k=3,bc="extrapolate")
                SF[:,iu,1] = itpDL(va)
            end
        end  # for iu
        # Sf1  , ok
        Sf[:,:,1] = fvu[:,:,isp]
        # Sf2 , ok  = DL[:,:,1] * fvu[nvgauss,:,isp]
        SF[:,:,2] = dH
        Sf[:,:,2] = df
        # Sf3 , ok
        SF[:,:,3] = sin2 .* (Hvu[:,:,isp] * DP[:,:,1])
        Sf[:,:,3] = fvu[:,:,isp] * DP[:,:,1]
        # Sf5 , ok = DL[:,:,2] * fvu[nvg,:,isp]    # may cause instability
        SF[:,:,4] = ddG
        Sf[:,:,4] = ddf
        # Sf6  , ok
        SF[:,:,5] = sin2 .* (Gvu[:,:,isp] * DP[:,:,1])
        Sf[:,:,5] = Sf[:,:,3]
        # A = (Gvu[:,:,isp]./vremesh.^nn * DP[:,:,1]) .* (fvu[:,:,isp]./vremesh.^nn * DP[:,:,1])
        # Sf8  , ok
        SF[:,:,6] = sin2 .* ((Gvu[:,:,isp] ./va - dG) * DP[:,:,1])
        Sf[:,:,6] = Sf[:,:,3] ./v - df * DP[:,:,1]
        # Sf9  , ok
        SF[:,:,7] = dG + (Gvu[:,:,isp] *(sin2 .*DP[:,:,2] - mu .* DP[:,:,1] )) ./va
        Sf[:,:,7] = df + (fvu[:,:,isp] *(sin2 .*DP[:,:,2] - mu .* DP[:,:,1] )) ./v
        # Sf13 , ok
        SF[:,:,8] = dG - Gvu[:,:,isp] * (mu .* DP[:,:,1] ) ./va
        Sf[:,:,8] = df - (fvu[:,:,isp] * (mu .* DP[:,:,1] )) ./v
        return Sf, SF
    end # if = 1
end


# for i = 4 # 1:nSf
# #     # display(plot(transpose(Sf[:,:,5])))
#     display(plot(va,SF[:,:,i],label=["isF=",i],xscale=:log))
#     display(plot(v,Sf[:,:,i],label=["isf=",i],xscale=:log))
# end

# # non-linerized model for axisymmetric distribution fL0[nv,nL,nspices]
# # m == 0
# function SfFL0(nspices::Int,isp::Int,nSf::Int,nv1::Int,ℓM1::Int,LM::AbstractVector,vth::AbstractVector,
#     m0::AbstractVector,v::AbstractVector,μ::AbstractVector,DP::AbstractArray{T4,N},
#     fvu::AbstractArray{T3,N},Hvu::AbstractArray{T3,N},Gvu::AbstractArray{T3,N}) where {T,T3,T4,N}
#
#     # dva_c = 1e-15  # = abs(vbath - 1), whether vath = vbth
#     kSpline1D = 3  # =1:5 for Dierckx.Spline1D
#     λ = 0.001
#     Sf = zeros(BigFloat,nv1,ℓM1,nSf)
#     SF = zeros(BigFloat,nv1,ℓM1,nSf)
#     mu = transpose(μ)
#     sin2 = (1 .-mu.^2)
#     vath = vth[isp]
#     nsp_vec = 1:nspices
#     ############ SF = Mvn * XLm * Mun , X = F, H ,G, X = X(v̂)
#     nspF = nsp_vec[nsp_vec .≠ isp]
#     if length(nspF) == 1
#         iFv = nspF[1]
#         mM = m0[isp] / m0[iFv]
#         vbth = vth[iFv]
#         vabth = vath / vbth
#         dva = abs(vabth - 1)  # abs(va[nv1] - v[nv1]) / v[nv1]
#         va = v * vabth        # 𝓋̂ = v/vbth = (v̂ * vath) / vbth
#         #### for the derivatives of dⁿH/dvⁿ and dⁿG/dvⁿ, H = H(𝓋̂ )
#         ##### SF2 = D^(1) * Hvu[:,:,isp] , inapplicable for H and dH = zeros(nv1,ℓM1)
#         # ###### dG = DL[:,:,1] * Gvu[:,:,isp]
#         ####### Sf # df = zeros(nv1,ℓM1)  #
#         df = zeros(nv1,ℓM1)
#         ddf = zeros(nv1,ℓM1)
#         dH = zeros(nv1,ℓM1)  #
#         dG = zeros(nv1,ℓM1)
#         ddG = zeros(nv1,ℓM1)
#         fvu_c = 1e-4         # Critical vaule of f(v̂→0,u) for df/dv due to boundary error of f(v,ℓ) * Mμ
#         for iu in 1:ℓM1
#             spl = Spline1D(v,fvu[:,iu,isp])
#             df[:,iu] = derivative(spl,v)
#             if fvu[4,iu,isp] < fvu_c
#                 df[1:3,iu] .= df[4,iu]
#             end
# # λ=0.001
# # spl = fit(SmoothingSpline, Float64.(vremesh), Float64.(a), λ) # bigger λ means the less freedom
# # dfs = SmoothingSplines.predict(spl)
# # plot(vremesh,a,xscale=:log)
# # plot!(vremesh,dfs,line=(3,:dot),xscale=:log)
#         # pf = plot(Float64.(v),df[:,iu],label=["isp=",isp],xscale=:log)
#     # spl = fit(SmoothingSpline, Float64.(v), Float64.(df[:,iu]), λ) # bigger λ means the less freedom
#     # df[:,iu] = SmoothingSplines.predict(spl) # fitted vector
#             spl = Spline1D(v,df[:,iu])
#             ddf[:,iu] = derivative(spl,v)
#     # spl = fit(SmoothingSpline, Float64.(v),Float64.(ddf[:,iu]), λ) # bigger λ means the less freedom
#     # ddf[:,iu] = SmoothingSplines.predict(spl) # fitted vector
#             # display(plot(fvu[:,1:iu,isp],label=["L=",iu-1]))
#     # pfs = plot!(Float64.(v),df[:,iu],label=["ispS=",isp],line=(3,:dot),xscale=:log)
#     # if iu == 1
#     #     display(pfs)
#     # end
#             # display(plot(pf,pfs,layout=(2,1)))
#         end
#         for iu in 1:ℓM1
#             spl = Spline1D(va,Hvu[:,iu,isp])
#             dH[:,iu] = derivative(spl,va)
#             # dh = diff(Hvu[:,iu,isp]) ./ diff(va)
#             # dH[2:end,iu] = dh
#             # dH[1,iu] = dh[1]
#             ##
#             spl = Spline1D(va,Gvu[:,iu,isp])
#             dG[:,iu] = derivative(spl,va)
#     # spl = fit(SmoothingSpline, Float64.(v), Float64.(dG[:,iu]), λ) # bigger λ means the less freedom
#     # dG[:,iu] = SmoothingSplines.predict(spl) # fitted vector
#             spl = Spline1D(va,dG[:,iu])
#             ddG[:,iu] = derivative(spl,va;nu=1)
#     # spl = fit(SmoothingSpline, Float64.(v), Float64.(ddG[:,iu]), λ) # bigger λ means the less freedom
#     # ddG[:,iu] = SmoothingSplines.predict(spl) # fitted vector
#             # SF1  , ok    # Interpolate F(v̂ᵦ) to F(𝓋̂ )
#             # if dva > dva_c  # else no intepolation is necessary
#                 if va[nv1] ≤ v[nv1]  # vath ≥ vbth
#                     itpDL = QuadraticInterpolation(fvu[:,iu,iFv],v)
#                     SF[:,iu,1] = itpDL.(va)
#                 else
#                     itpDL = Spline1D(v,fvu[:,iu,iFv];k=kSpline1D,bc="extrapolate")
#                     SF[:,iu,1] = itpDL(va)
#                 end
#             # end # dva
#         end  # for iu
#         #
#         # Sf1  , ok
#         Sf[:,:,1] = fvu[:,:,isp]
#         # Sf2 , ok  = DL[:,:,1] * fvu[nvgauss,:,isp]
#         SF[:,:,2] = dH
#         Sf[:,:,2] = df
#         # Sf3 , ok
#         SF[:,:,3] = sin2 .* (Hvu[:,:,isp] * DP[:,:,1])
#         Sf[:,:,3] = fvu[:,:,isp] * DP[:,:,1]
#         # Sf5 , ok = DL[:,:,2] * fvu[nvg,:,isp]    # may cause instability
#         SF[:,:,4] = ddG
#         Sf[:,:,4] = ddf
#         # Sf6  , ok
#         SF[:,:,5] = sin2 .* (Gvu[:,:,isp] * DP[:,:,1])
#         Sf[:,:,5] = Sf[:,:,3]
#         # A = (Gvu[:,:,isp]./vremesh.^nn * DP[:,:,1]) .* (fvu[:,:,isp]./vremesh.^nn * DP[:,:,1])
#         # Sf8  , ok
#         SF[:,:,6] = sin2 .* ((Gvu[:,:,isp] ./va - dG) * DP[:,:,1])
#         Sf[:,:,6] = Sf[:,:,3] ./v - df * DP[:,:,1]
#         # Sf9  , ok
#         SF[:,:,7] = dG + (Gvu[:,:,isp] *(sin2 .*DP[:,:,2] - mu .* DP[:,:,1] )) ./va
#         Sf[:,:,7] = df + (fvu[:,:,isp] *(sin2 .*DP[:,:,2] - mu .* DP[:,:,1] )) ./v
#         # Sf13 , ok
#         SF[:,:,8] = dG - Gvu[:,:,isp] * (mu .* DP[:,:,1] ) ./va
#         Sf[:,:,8] = df - (fvu[:,:,isp] * (mu .* DP[:,:,1] )) ./v
#
# # for i = 4 # 1:nSf
# # #     # display(plot(transpose(Sf[:,:,5])))
# #     display(plot(va,SF[:,:,i],label=["isF=",i],xscale=:log))
# #     display(plot(v,Sf[:,:,i],label=["isf=",i],xscale=:log))
# # end
# #
# # for i in 1:nSf
# #     # display(plot(transpose(Sf[:,:,5])))
# #     # display(plot(SF[:,:,i].*Sf[:,:,i],label=["isf=",i]))
# #     # display(plot(SF[:,:,i],label=["isf=",i]))
# #     pf = plot(Sf[:,:,i],label=["isf=",i])
# #     pF = plot(SF[:,:,i],label=["isF=",i])
# #     display(plot(pf,pF,layout=(2,1)))
# # end
#
# # for i in [1,2,4,8] # 1:nSf
# #     # display(plot(transpose(Sf[:,:,5])))
# #     display(plot(va,SF[:,:,i],label=["isf=",i],xscale=:log))
# # end
#
#     return Sf, SF

    # display(plot(Sf[:,:,1],label=["f=",isp]))
    # display(plot(ddf,label=["ddf=",isp]))
    # # Sf1  , ok
    # SF[:,:,1] = SF[:,:,1] .* fvu[:,:,isp]
    # # Sf[:,:,1] =  fvu[:,:,isp]
    # # Sf2 , ok  = DL[:,:,1] * fvu[nvgauss,:,isp]
    # SF[:,:,2] = dH .* df
    # # Sf[:,:,2] = df
    # # Sf3 , ok
    # A3 = fvu[:,:,isp] * DP[:,:,1]
    # SF[:,:,3] = A3 .* (sin2 .* (Hvu[:,:,isp] * DP[:,:,1]))
    # # Sf5 , ok = DL[:,:,2] * fvu[nvgauss,:,isp]
    # SF[:,:,4] = ddG .* ddf
    # # Sf[:,:,4] = ddf
    # # Sf6  , ok
    # SF[:,:,5] = A3 .* sin2 .* (Gvu[:,:,isp] * DP[:,:,1])
    # # Sf[:,:,5] = Sf[:,:,3]
    # # Sf8  , ok
    # A1 = A3 ./v - df * DP[:,:,1]
    # SF[:,:,6] = A1 .* sin2 .* ((Gvu[:,:,isp] ./va - dG) * DP[:,:,1])
    # # Sf9  , ok
    # A1 = df + (fvu[:,:,isp] *(sin2 .*DP[:,:,2] - mu .* DP[:,:,1] )) ./v
    # SF[:,:,7] = A1 .* (dG + (Gvu[:,:,isp] *(sin2 .*DP[:,:,2] - mu .* DP[:,:,1] )) ./va)
    # # Sf13 , ok
    # A1 = df - (fvu[:,:,isp] * (mu .* DP[:,:,1] )) ./v
    # SF[:,:,8] = A1 .* (dG - Gvu[:,:,isp] * (mu .* DP[:,:,1] ) ./va)
    #     return  SF




function collisionL0(nspices::Int,nSf::Int,nSdft::Int,nv1::Int,ℓM1::Int,LM::AbstractVector,
    vth::AbstractVector,m0::AbstractVector,Zq::AbstractVector,n0::AbstractVector,v::AbstractVector,
    μ::AbstractVector,DP::AbstractArray{T4,N},fvu::AbstractArray{T3,N},Hvu::AbstractArray{T3,N},
    Gvu::AbstractArray{T3,N},Mμ::AbstractArray{T2,N2}) where {T,T2,T3,T4,N,N2}

    mu = transpose(μ)
    sin2 = 1 .- mu.^2
    v2 = v.^2
    ######################## δft = Γa ∑ᵢSᵢ,  only the summation output
    if nSdft == 1
        δft = zeros(BigFloat,nv1,ℓM1,nspices)
        cs = ones(6)
        cs = ones(6)
        #### δfₗ/δt = δft * Mμ
        for isp in 1:nspices
            vath = vth[isp]
            nsp_vec = 1:nspices
            ############ SF = Mvn * XLm * Mun , X = F, H ,G, X = X(v̂)
            nspF = nsp_vec[nsp_vec .≠ isp]
            # if length(nspF) == 1
                iFv = nspF[1]
                mM = m0[isp] / m0[iFv]
                nb = n0[iFv]
                vbth = vth[iFv]
                vabth = vath / vbth

                qma = qma0 * Zq[isp] * Zq[iFv] / (m0[isp]/ Da)
                lnAab = lnA()
                Γa = 4π * qma^2 * lnAab
            #### coefficients
            # CfL = n0[isp] / π^1.5 / vath^3  # Normalized coefficients of fₗₘ
            CFL = nb / π^1.5 / vbth^3  # Normalized coefficients of Fₗₘ
            CHL = nb * (1+mM) * 4/√π /vbth  # Normalized coefficients of Hₗₘ
            CGL = nb * 4/√π * vbth           # Normalized coefficients of Gₗₘ
            CfFL = Γa * CFL * 4π * mM
            CfHL = Γa * CHL * (1 - mM) / (1 + mM)
            CfGL = Γa * CGL * (1 // 2)
            va3vb1 = vath^3 * vbth
            ###
            cs[1] = CfFL
            cs[2] = CfHL / (vath * vbth)
            cs[3] = CfHL /vath^2         #     ./ v2
            cs[4] = CfGL / (vath * vbth)^2
            cs[5] = CfGL /vath^4         #    ./ v2.^2
            cs[6] =  CfGL/va3vb1     # = cs[7] = cs[8]     ./ v2
            # println(cs)
            #######
            Sf, SF = SfFL0(nspices,isp,nSf,nv1,ℓM1,LM,vth,m0,v,μ,DP,fvu,Hvu,Gvu)  # vabth
            # for i in 1:nSf
            #     # δft[:,:,i,isp] =  Sf[:,:,i] .* SF[:,:,i] * Mμ
            #     δft[:,:,i,isp] = (Sf[:,:,i] .* SF[:,:,i])
            # end
            ###
            for i in [1,2,4]
                δft[:,:,isp] += cs[i] * (Sf[:,:,i] .* SF[:,:,i])
            end
            i = 3
            δft[:,:,isp] += cs[i] ./ v2.^2 .* (Sf[:,:,i] .* SF[:,:,i])
            i = 5   #  ./ v2.^2 however the numerical errors blows up the data when v̂ ≪ 1
                    #  ./v.^(2~3) maybe is a resonable which is right qualitatively without the absolute value.
                    #  δft[0,:,0,isp] ∝ 1/v^3
            δft[:,:,isp] += cs[5]  ./ v2 ./v .* (Sf[:,:,i] .* SF[:,:,i]) * Mμ
            for i in [6,7,8]
                δft[:,:,isp] += cs[6] ./ v2 .* (Sf[:,:,i] .* SF[:,:,i])
            end
            ## Project δft(v̂,μ,isp) to δfₗ/δt(v̂,ℓ,isp)
            δft[:,:,isp] = δft[:,:,isp] * Mμ  # δfₗ/δt
#     pF = plot(v, (CfL*cs[1]/Γa*Sf[:,1,1] .*  SF[:,1,1]),xscale=:log,label=["F0,isp=",isp])
# # pF = plot(v, (CfL*Sf[:,1,1]),xscale=:log,label=["F0,isp=",isp])
#     display(pF)
    # for i in [1,2,4,8]  #1:nSf
    #     # display(plot(v, (Sf[:,:,i] .* SF[:,:,i]),xscale=:log,label=["isf=",i]))
    #     display(plot(v, (Sf[:,:,i]),xscale=:log,label=["isf=",i]))
    # end
    # i = 1
    # p1 = plot(v, (cs[i]/Γa * (Sf[:,:,i] .* SF[:,:,i])),xscale=:log,label=["isf=",1])
    # i = 2
    # p2 = plot(v, (cs[i]/Γa * (Sf[:,:,i] .* SF[:,:,i])),xscale=:log,label=["isf=",2])
    # i = 4
    # p4 = plot(v, (cs[i]/Γa * (Sf[:,:,i] .* SF[:,:,i])),xscale=:log,label=["isf=",4])
    # i = 8
    # p8 = plot(v, (cs[6]/Γa ./ v2 .* (Sf[:,:,i] .* SF[:,:,i])),xscale=:log,label=["isf=",8])
    # display(plot(p1,p2,p4,p8,layout=(2,2),legend=false))
        end  # for isp
        return δft
    else  ######################### δft(8)
        δft = zeros(BigFloat,nv1,ℓM1,nSf,nspices)
        cs = ones(6)
        #### δfₗ/δt = δft * Mμ
        for isp in 1:nspices
            vath = vth[isp]
            nsp_vec = 1:nspices
            ############ SF = Mvn * XLm * Mun , X = F, H ,G, X = X(v̂)
            nspF = nsp_vec[nsp_vec .≠ isp]
            # if length(nspF) == 1
                iFv = nspF[1]
                mM = m0[isp] / m0[iFv]
                nb = n0[iFv]
                vbth = vth[iFv]
                vabth = vath / vbth

                qma = qma0 * Zq[isp] * Zq[iFv] / (m0[isp]/ Da)
                lnAab = lnA()
                Γa = 4π * qma^2 * lnAab
            #### coefficients
            # CfL = n0[isp] / π^1.5 / vath^3  # Normalized coefficients of fₗₘ
            CFL = nb / π^1.5 / vbth^3  # Normalized coefficients of Fₗₘ
            CHL = nb * (1+mM) * 4/√π /vbth  # Normalized coefficients of Hₗₘ
            CGL = nb * 4/√π * vbth           # Normalized coefficients of Gₗₘ
            CfFL = Γa * CFL * 4π * mM
            CfHL = Γa * CHL * (1 - mM) / (1 + mM)
            CfGL = Γa * CGL * (1 // 2)
            va3vb1 = vath^3 * vbth
            ###
            cs[1] = CfFL
            cs[2] = CfHL / (vath * vbth)
            cs[3] = CfHL /vath^2         #     ./ v2
            cs[4] = CfGL / (vath * vbth)^2
            cs[5] = CfGL /vath^4         #    ./ v2.^2
            cs[6] =  CfGL/va3vb1     # = cs[7] = cs[8]     ./ v2
            # println(cs)
            #######
            Sf, SF = SfFL0(nspices,isp,nSf,nv1,ℓM1,LM,vth,m0,v,μ,DP,fvu,Hvu,Gvu)
            # for i in 1:nSf
            #     # δft[:,:,i,isp] =  Sf[:,:,i] .* SF[:,:,i] * Mμ
            #     δft[:,:,i,isp] = (Sf[:,:,i] .* SF[:,:,i])
            # end
            ###
            for i in [1,2,4]
                δft[:,:,i,isp] = cs[i] * (Sf[:,:,i] .* SF[:,:,i]) * Mμ
            end
            i = 3
            δft[:,:,i,isp] = cs[i] ./ v2 .* (Sf[:,:,i] .* SF[:,:,i]) * Mμ
            i = 5   #  ./ v2.^2 however the numerical errors blows up the data when v̂ ≪ 1
                    #  ./v.^(2~3) maybe is a resonable which is right qualitatively without the absolute value.
                    #  δft[0,:,0,isp] ∝ 1/v^3
            δft[:,:,i,isp] = cs[5]  ./ v2.^2 .* (Sf[:,:,i] .* SF[:,:,i]) * Mμ
            for i in [6,7,8]
                δft[:,:,i,isp] = cs[6] ./ v2 .* (Sf[:,:,i] .* SF[:,:,i]) * Mμ
            end
            #
            # p1 = plot(v,δft[:,:,1,isp],xscale=:log,label=["isf=",1])
            # p2 = plot(v,δft[:,:,2,isp],xscale=:log,label=["isf=",2])
            # p3 = plot(v,δft[:,:,3,isp],xscale=:log,label=["isf=",3])
            # p4 = plot(v,δft[:,:,4,isp],xscale=:log,label=["isf=",4])
            # p5 = plot(v,δft[:,:,5,isp],xscale=:log,label=["isf=",5])
            # p6 = plot(v,δft[:,:,6,isp],xscale=:log,label=["isf=",6])
            # p7 = plot(v,δft[:,:,7,isp],xscale=:log,label=["isf=",7])
            # p8 = plot(v,δft[:,:,8,isp],xscale=:log,label=["isf=",8])
            # display(plot(p1,p2,p4,p7,layout=(2,2)))
            # # δft = ∑ᵢSᵢ
            # for i = 2:nSf
            #     δft[:,:,1,isp] = δft[:,:,1,isp] + δft[:,:,i,isp]
            # end
            # display(plot(v,δft[:,:,1,isp],xscale=:log,label=["isf=",0]))
        end  # for isp
        return δft[:,:,1,:]
    end # nSdft == 1
end
