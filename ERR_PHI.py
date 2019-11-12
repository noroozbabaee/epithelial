import math
import numpy as np
from scipy.optimize import *
import sys

sys.path.append('../sfcpt/')
from PCT_GLOB import *

Scale=1.e6
np.log(sys.float_info.min)
np.log(sys.float_info.min * sys.float_info.epsilon)

def LMMSC(C_a, C_b):
    #''' Logarithm mean membrane solute concentration '''
    import math
    math.log(sys.float_info.min)
    math.log(sys.float_info.min * sys.float_info.epsilon)
    if C_a/C_b > 0 and C_a-C_b != 0:
        return (C_a-C_b)/(math.log10(C_a/C_b))
    else:
        return C_b


def NAH(CI_H, CI_NA, CI_NH4, CM_H, CM_NA, CM_NH4, param_NAH):
    if param_NAH==0:
        return[0,0,0]
    else:
        CXT = 1.00
        KNAH_NA = 0.3000e-01
        KNAH_H = 0.7200e-07
        KNAH_NH4 = 0.2700e-01
        KNAH_I = 0.1000e-05
        # DATA PNAH/.8000D+04,.2400D+04,.8000D+04,.2000D+01,.0000D+00/

        PNAH_NA = 0.8000e+04
        PNAH_H = 0.2400e+04
        PNAH_NH4 = 0.8000e+04
        PNAH_m = 0.2000e+01
        PNAH_M = 0.0000

        # TRANSLATE CONCENTRATIONS TO THE NAH MODEL
        PSNAH_NA = PNAH_NA * (PNAH_M * CI_H + PNAH_m * KNAH_I) / (CI_H + KNAH_I)
        PSNAH_H = PNAH_H * (PNAH_M * CI_H + PNAH_m * KNAH_I) / (CI_H + KNAH_I)
        PSNAH_NH4 = PNAH_NH4 * (PNAH_M * CI_H + PNAH_m * KNAH_I) / (CI_H + KNAH_I)
        CISNAH_NA = CI_NA / KNAH_NA
        CMSNAH_NA = CM_NA / KNAH_NA
        CISNAH_H = CI_H / KNAH_H
        CMSNAH_H = CM_H / KNAH_H
        CISNAH_NH4 = CI_NH4 / KNAH_NH4
        CMSNAH_NH4 = CM_NH4 / KNAH_NH4
        DELTA_I = 1.0
        DELTA_M = 1.0
        DELTA_I = DELTA_I + CISNAH_NA + CISNAH_H + CISNAH_NH4
        DELTA_M = DELTA_M + CMSNAH_NA + CMSNAH_H + CMSNAH_NH4
        EPSILON_I = PSNAH_NA * CISNAH_NA + PSNAH_H * CISNAH_H + PSNAH_NH4 * CISNAH_NH4
        EPSILON_M = PSNAH_NA * CMSNAH_NA + PSNAH_H * CMSNAH_H + PSNAH_NH4 * CMSNAH_NH4
        SIGMA = DELTA_I * EPSILON_M + DELTA_M * EPSILON_I
        JNAH_NA = 0.0
        JNAH_H = 0.0
        JNAH_NH4 = 0.0
        JNAH_NA = JNAH_NA + (PSNAH_NA * PSNAH_H * CXT / SIGMA) * (CMSNAH_NA * CISNAH_H - CISNAH_NA * CMSNAH_H)
        +  (PSNAH_NA * PSNAH_NH4 * CXT / SIGMA) * (CMSNAH_NA * CISNAH_NH4 - CISNAH_NA * CMSNAH_NH4)
        JNAH_H = JNAH_H + (PSNAH_H * PSNAH_NA * CXT / SIGMA) * (CMSNAH_H * CISNAH_NA - CISNAH_H * CMSNAH_NA)
        +  (PSNAH_H * PSNAH_NH4 * CXT / SIGMA) * (CMSNAH_H * CISNAH_NH4 - CISNAH_H * CMSNAH_NH4)
        JNAH_NH4 = JNAH_NH4 + (PSNAH_NH4 * PSNAH_NA * CXT / SIGMA) * (CMSNAH_NH4 * CISNAH_NA - CISNAH_NH4 * CMSNAH_NA)
        +  (PSNAH_NH4 * PSNAH_H * CXT / SIGMA) * (CMSNAH_NH4 * CISNAH_H - CMSNAH_H * CISNAH_NH4)
        JNAH_NA_Max = CXT * PSNAH_NA * PSNAH_H / (PSNAH_NA + PSNAH_H)
        return[JNAH_NA, JNAH_H, JNAH_NH4]


def EBUF(C_H, PK , Ca, Cb):
    def LH (C_H):
        if C_H > 0 and C_H != 0:
           return -1. * math.log10(C_H)
        else:
           return -1. * math.log10(0.002)

    LCH = LH(C_H)
    if (Ca / Cb) > 0 and (Ca/Cb) != 0:
        return LCH - PK - math.log10(Ca/Cb)
    else:
        return LCH - PK


def KCL_IS(CI_K, CS_K, CI_CL, CS_CL, Z_K, Z_CL,  VI, VS, AIS, LIS_KCL, param_KCL_IS):
    if param_KCL_IS==0:
        return[0,0]
    else:
        XI_K = EPS(CI_K, Z_K, VI)
        XI_CL = EPS(CI_CL, Z_CL, VI)
        XS_K = EPS(CS_K, Z_K, VS)
        XS_CL = EPS(CS_CL, Z_CL, VS)
        K_IS_KCL = LIS_KCL * AIS * (XI_K - XS_K + XI_CL - XS_CL)
        CL_IS_KCL = LIS_KCL * AIS * (XI_K - XS_K + XI_CL - XS_CL)
    return[K_IS_KCL, CL_IS_KCL]

def EPS(C, Z, V):
    if C > 0 and C != 0:
        return RTE * math.log10(C) + Z * F * V * 1.e-6
    else:
        return Z * F * V * 1.e-6


def CLHCO3_MI(CM_CL, CI_CL, CM_HCO3, CI_HCO3, Z_CL, Z_HCO3, VM, VI, AMI, LMI_CLHCO3, param_CLHCO3_MI):
    if param_CLHCO3_MI==0:
        return[0,0]
    else:
        XM_CL = EPS(CM_CL, Z_CL, VM)
        XM_HCO3 = EPS(CM_HCO3, Z_HCO3, VM)
        XI_CL = EPS(CI_CL, Z_CL, VI)
        XI_HCO3 = EPS(CI_HCO3, Z_HCO3, VI)
        CL_MI_CLHCO3 = LMI_CLHCO3 * AMI * (XM_CL - XI_CL - XM_HCO3 + XI_HCO3)
        HCO3_MI_CLHCO3 = - LMI_CLHCO3 * AMI * (XM_CL - XI_CL - XM_HCO3 + XI_HCO3)
    return[CL_MI_CLHCO3, HCO3_MI_CLHCO3]



def NA_CLHCO3_IS(CI_NA, CS_NA, CI_CL, CS_CL, CI_HCO3, CS_HCO3, Z_NA, Z_CL, Z_HCO3, VI, VS, AIS, LIS_NA_CLHCO3, param_NA_CLHCO3_IS):
    if param_NA_CLHCO3_IS==0:
        return[0,0,0]
    else:
        XI_NA = EPS(CI_NA, Z_NA, VI)
        XI_CL = EPS(CI_CL, Z_CL, VI)
        XI_HCO3 = EPS(CI_HCO3, Z_HCO3, VI)
        XS_NA = EPS(CS_NA, Z_NA, VS)
        XS_CL = EPS(CS_CL, Z_CL, VS)
        XS_HCO3 = EPS(CS_HCO3, Z_HCO3, VS)
        NA_NA_CLHCO3 = + AIS*LIS_NA_CLHCO3 * (XI_NA - XS_NA  - XI_CL  + XS_CL  + 2 * (XI_HCO3  - XS_HCO3))
        CL_NA_CLHCO3 = - AIS*LIS_NA_CLHCO3 * (XI_NA  - XS_NA - XI_CL  + XS_CL  + 2 * (XI_HCO3 - XS_HCO3 ))
        HCO3_NA_CLHCO3 = 2 *AIS* LIS_NA_CLHCO3 * (XI_NA  - XS_NA  - XI_CL + XS_CL  + 2 * (XI_HCO3 - XS_HCO3))
        return[NA_NA_CLHCO3, CL_NA_CLHCO3, HCO3_NA_CLHCO3]


def CLHCO2_MI(CM_CL, CI_CL, CM_HCO2, CI_HCO2, Z_CL, Z_HCO2, VM, VI, AMI, LMI_CLHCO2, param_CLHCO2_MI):
    if param_CLHCO2_MI==0:
        return[0,0]
    else:
        XM_CL = EPS(CM_CL, Z_CL, VM)
        XM_HCO2 = EPS(CM_HCO2, Z_HCO2, VM)
        XI_CL = EPS(CI_CL, Z_CL, VI)
        XI_HCO2 = EPS(CI_HCO2, Z_HCO2, VI)
        CL_MI_CLHCO2 = LMI_CLHCO2 * AMI * (XM_CL - XI_CL - XM_HCO2 + XI_HCO2)
        HCO2_MI_CLHCO2 = - LMI_CLHCO2 * AMI * (XM_CL - XI_CL - XM_HCO2 + XI_HCO2)
        return[CL_MI_CLHCO2, HCO2_MI_CLHCO2]


def SGLT_MI(CM_NA, CI_NA, CM_GLUC, CI_GLUC, Z_GLUC, Z_NA,VM,VI, AMI, LMI_NAGLUC, param_SGLT_MI):
    if param_SGLT_MI==0:
        return[0,0]
    else:
        XM_NA  = EPS(CM_NA, Z_NA, VM)
        XM_GLUC  = EPS(CM_GLUC, Z_GLUC, VM)
        XI_NA  = EPS(CI_NA, Z_NA, VI)
        XI_GLUC  = EPS(CI_GLUC, Z_GLUC, VI)
        NA_MI_NAGLUC = LMI_NAGLUC* AMI * (XM_NA - XI_NA  + XM_GLUC - XI_GLUC )
        GLUC_MI_NAGLUC = LMI_NAGLUC * AMI * (XM_NA  - XI_NA  + XM_GLUC  - XI_GLUC )
        return[NA_MI_NAGLUC , GLUC_MI_NAGLUC]


def NAK_ATP(CI_K, CS_K, CI_NA, CS_NA, CE_K, CE_NH4, param_NAK_ATP):
    if param_NAK_ATP==0:
        return[0,0,0]
    else:
        KNPN = 0.0002 * (1.0 + CI_K / .00833)
    # SODIUM AFFINITY
        KNPK = 0.0001 * (1.0 + CS_NA / .0185)
    # ATPASE TRANSPORTER FLUX IN IS MEMBRANE
        ATIS_NA = NP*(CI_NA / (KNPN + CI_NA))**3*(CS_K/(KNPK + CS_K))**2
    # ALLOW FOR COMPETITION BETWEEN K+ AND NH4+

        ATIS_K = -ATIS_NA * 0.667 * CE_K / (CE_K + CE_NH4 / KNH4)
        ATIS_NH4 = -ATIS_NA * 0.667 * CE_NH4 / (KNH4 * CE_K + CE_NH4)
        return[ATIS_NA, ATIS_K, ATIS_NH4]


def AT_MI_H(CM_H, CI_H, VM, VI, Z_H, param_AT_MI_H):
    XM_H = EPS(CM_H, Z_H, VM)
    XI_H = EPS(CI_H, Z_H, VI)
    if param_AT_MI_H==0:
        return[0]
    elif XIHP * (XM_H - XI_H - XHP) > 0 and math.exp(-(XIHP * (XM_H - XI_H - XHP))) != 0:
        return -LHP / (1.0 + 1/math.exp(-(XIHP * (XM_H - XI_H - XHP))))
    else:
        return -LHP / (1.0 + math.exp(XIHP * (XM_H - XI_H - XHP)))


def NAH2PO4_MI(CM_NA, CI_NA, CM_H2PO4, CI_H2PO4,Z_NA , Z_H2PO4,  VM, VI, AMI, LMI_NAH2PO4, param_NAH2PO4_MI):
    if param_NAH2PO4_MI==0:
      return[0,0]
    else:
      XM_NA = EPS(CM_NA, Z_NA, VM)
      XM_H2PO4 = EPS(CM_H2PO4, Z_H2PO4, VM)
      XI_NA = EPS(CI_NA, Z_NA, VI)
      XI_H2PO4 = EPS(CI_H2PO4, Z_H2PO4, VI)
      NA_MI_NAH2PO4 = LMI_NAH2PO4 * AMI * (XM_NA - XI_NA + XM_H2PO4 - XI_H2PO4)
      H2PO4_MI_NAH2PO4 = LMI_NAH2PO4 * AMI * (XM_NA - XI_NA + XM_H2PO4 - XI_H2PO4)
    return[NA_MI_NAH2PO4, H2PO4_MI_NAH2PO4]


def NAHCO3_IS(CI_NA, CS_NA, CI_HCO3, CS_HCO3, Z_NA, Z_HCO3, VI, VS, AIS, LIS_NAHCO3,param_NAHCO3_IS):
    if param_NAHCO3_IS==0:
        return[0,0]
    else:
        XI_NA = EPS(CI_NA, Z_NA, VI)
        XI_HCO3 = EPS(CI_HCO3, Z_HCO3, VI)
        XS_NA = EPS(CS_NA, Z_NA, VS)
        XS_HCO3 = EPS(CS_HCO3, Z_HCO3, VS)
        NA_IS_NAHCO3 = LIS_NAHCO3 * AIS * (XI_NA - XS_NA + 3 * (XI_HCO3 - XS_HCO3))
        HCO3_IS_NAHCO3 = 3 * LIS_NAHCO3 * AIS * (XI_NA - XS_NA + 3 * (XI_HCO3 - XS_HCO3))
        return[NA_IS_NAHCO3, HCO3_IS_NAHCO3]

def GOLDMAN(hab, A, Zab, Va, Vb, ca, cb, param_GOLDMAN):
    Zab = (Zab*F*(Va-Vb)*1.e-6)/RTE

    if param_GOLDMAN==0:
        return[0]
    elif Zab==0 or Va==Vb:
        return hab * A * (ca - cb)
    elif Zab>0:
        return hab * A * Zab * (ca - cb * math.exp(-Zab)) / (1 - math.exp(-Zab))
    else:
        return hab * A * Zab * (ca * math.exp(Zab) - cb) / (math.exp(Zab) - 1)
        # normalized electrical potential difference




def LMMS(C_a, C_b):
    #''' Logarithm mean membrane solute concentration '''
    import math
    math.log(sys.float_info.min)
    math.log(sys.float_info.min * sys.float_info.epsilon)
    if C_a/C_b > 0 and C_a-C_b != 0:
        return (C_a-C_b)/(math.log10(C_a/C_b))
    else:
        return C_b


def EPS(C, Z, V):
    #'''Electrochemical Potential of Species i'''
    import math
    math.log(sys.float_info.min)
    math.log(sys.float_info.min * sys.float_info.epsilon)
    if C > 0 and C!=0:
        return RTE * math.log10(C) + Z * F * V * 1.e-6
    else:
        return Z*F*V*1.e-6


def Zab(Z, Va, Vb):
        return Z * F * (Va - Vb) * 1.e-6 / RTE

def PHIScale(PHI, Scale):
    return PHI * Scale
import sys

sys.path.append('../epithelia/')
from PCT_GLOB import *

def EQS(i, GUESS, flag):
    #COMPUTE THE RELEVANT VOLUMES

    AE[t]= AE0 * (MUA * (PE[t] - PM))
    AE[t]= AE0 if AE[t] < AE0 else AE[t]
    print(AE0,'that is working')
    CHVL[t]= CHVL0 * (1.0 + MUV * (PE[t] - PM))
    CHVL[t]= CHVL0 if CHVL[t] < CHVL0 else CHVL[t]

    L[t]= CHVL[t]

    LCHE = PKC + np.log10(CE_HCO3[t])-np.log10( CE_H2CO3[t] )
    CE_H[t]= 10. ** (-LCHE)

    # CLVL= CLVL0 * IMP0 / IMP
    # CLVL= CLVL0 if IMP== 0 else CLVL[t]
    #
    # L= L[t] + CLVL[t]
    # PI= PM
    #
    # LCHI = PKC + np.log10(CI_HCO3[t]) - np.log10(CI_H2CO3[t])
    # CI_H= 10. ** (-LCHI)
    for k in range(0):
        CLVL[k][t]= CLVL0[k] * IMP0[k] / IMP[k]
        CLVL[k][t]= CLVL0[k] if IMP[k] == 0 else CLVL[k][t]

        L[t]= L[t] + CLVL[k][t]
        PI[k][t]= PM

        LCHI[k] = PKC + np.log10(CI_HCO3[k][t]) - np.log10(CI_H2CO3[k][t])
        CI_H[k]= 10. ** (-LCHI[k])
    #COMPUTE THE MEMBRANE FLUXES
    # FEVM= AME * LPME * (PM - PE[t] - RT * IMPM) / RT
    # FEVS= AE[t] * LPES * (RT * IMPS + PE[t] - PS) / RT
    # FIVM= AMI * LPMI * (RT * IMP - RT * IMPM + PM- PI[t]) / RT
    # FIVS= AIS * LPIS * (PI[t] - PS + RT * IMPS - RT * IMP) / RT
    # JV= LPIS * AIE * (PI[t] - PE[t] - RT * IMP) / RT
    FEVM[t]= AME * LPME * (PM - PE[t] - RT * IMPM) / RT
    FEVS[t]= AE[t] * LPES * (RT * IMPS + PE[t] - PS) / RT
    for k in range(0):
        FIVM[k][t]= AMI[k] * LPMI[k] * (RT * IMP[k] - RT * IMPM + PM - PI[k][t]) / RT
        FIVS[k][t]= AIS[k] * LPIS[k] * (PI[k][t] - PS + RT * IMPS - RT * IMP[k]) / RT
        JV[k][t]= LPIS[k] * AIE[k] * (PI[k][t] - PE[t] - RT * IMP[k]) / RT

    FEVM[t]= FEVM[t] + AME * LPME * (
            SME_NA * (CE_NA[t] - CM_NA) + SME_K * (
            CE_K[t] - CM_K) + SME_CL * (
                    CE_CL[t] - CM_CL) + SME_HCO3 * (
                    CE_HCO3[t] - CM_HCO3)
            + SME_H2CO3 * (CE_H2CO3[t] - CM_H2CO3) + SME_CO2 * (
                    CE_CO2[t] - CM_CO2) + SME_HPO4 * (
                    CE_HPO4[t] - CM_HPO4) + SME_H2PO4 * (
                    CE_H2PO4[t] - CM_H2PO4)
            + SME_UREA * (CE_UREA[t] - CM_UREA) + SME_NH3 * (
                    CE_NH3[t] - CM_NH3) + SME_NH4 * (
                    CE_NH4[t] - CM_NH4) + SME_H * (
                    CE_H[t] - CM_H) + SME_HCO2 * (
                    CE_HCO2[t] - CM_HCO2)
            + SME_H2CO2 * (CE_H2CO2[t] - CM_H2CO2) + SME_GLUC * (
                    CE_GLUC[t] - CM_GLUC))


    FEVS[t]= FEVS[t] + AE[t] * LPES * (
            SES_NA * (CS_NA - CE_NA[t]) + SES_K * (
            CS_K - CE_K[t]) + SES_CL * (
                    CS_CL - CE_CL[t]) + SES_HCO3 * (
                    CS_HCO3 - CE_HCO3[t])
            + SES_H2CO3 * (CS_H2CO3 - CE_H2CO3[t]) + SES_CO2 * (
                    CS_CO2 - CE_CO2[t]) + SES_HPO4 * (
                    CS_HPO4 - CE_HPO4[t]) + SES_H2PO4 * (
                    CS_H2PO4 - CE_H2PO4[t])
            + SES_UREA * (CS_UREA - CE_UREA[t]) + SES_NH3 * (
                    CS_NH3 - CE_NH3[t]) + SES_NH4 * (
                    CS_NH4 - CE_NH4[t]) + SES_H * (
                    CS_H - CE_H[t]) + SES_HCO2 * (
                    CS_HCO2 - CE_HCO2[t])
            + SES_H2CO2 * (CS_H2CO2 - CE_H2CO2[t]) + SES_GLUC * (
                    CS_GLUC - CE_GLUC[t]))

    for k in range(0):
        FIVM[k][t]= FIVM[k][t] + AMI[k] * LPMI[k] * (
                SMI_NA[k] * (CI_NA[k][t] - CM_NA) + SMI_K[k] * (
                CI_K[k][t] - CM_K) + SMI_CL[k] * (
                        CI_CL[k][t] - CM_CL) + SMI_HCO3[k] * (
                        CI_HCO3[k][t] - CM_HCO3)
                + SMI_H2CO3[k] * (CI_H2CO3[k][t] - CM_H2CO3) + SMI_CO2[k] * (
                        CI_CO2[k][t] - CM_CO2) + SMI_HPO4[k] * (
                        CI_HPO4[k][t] - CM_HPO4) + SMI_H2PO4[k] * (
                        CI_H2PO4[k][t] - CM_H2PO4)
                + SMI_UREA[k] * (CI_UREA[k][t] - CM_UREA) + SMI_NH3[k] * (
                        CI_NH3[k][t] - CM_NH3) + SMI_NH4[k] * (
                        CI_NH4[k][t] - CM_NH4) + SMI_HCO2[k] * (
                        CI_HCO2[k][t] - CM_HCO2)
                + SMI_H2CO2[k] * (CI_H2CO2[k][t] - CM_H2CO2) + SMI_GLUC[k] * (
                        CI_GLUC[k][t] - CM_GLUC))

        FIVS[k][t] = FIVS[k][t] + AIS[k] * LPIS[k] * (
                SIS_NA[k] * (CS_NA - CI_NA[k][t]) + SIS_K[k] * (
                CS_K - CI_K[k][t]) + SIS_CL[k] * (
                        CS_CL - CI_CL[k][t]) + SIS_HCO3[k] * (
                        CS_HCO3 - CI_HCO3[k][t])
                + SIS_H2CO3[k] * (CS_H2CO3 - CI_H2CO3[k][t]) + SIS_CO2[k] * (
                        CS_CO2 - CI_CO2[k][t]) + SIS_HPO4[k] * (
                        CS_HPO4 - CI_HPO4[k][t]) + SIS_H2PO4[k] * (
                        CS_H2PO4 - CI_H2PO4[k][t])
                + SIS_UREA[k] * (CS_UREA - CI_UREA[k][t]) + SIS_NH3[k] * (
                        CS_NH3 - CI_NH3[k][t]) + SIS_NH4[k] * (
                        CS_NH4 - CI_NH4[k][t]) + SIS_H[k] * (
                        CS_H - CI_H[k][t]) + SIS_HCO2[k] * (
                        CS_HCO2 - CI_HCO2[k][t])
                + SIS_H2CO2[k] * (CS_H2CO2 - CI_H2CO2[k][t]) + SIS_GLUC[k] * (
                        CS_GLUC - CI_GLUC[k][t]))


        JV[k][t] = JV[k][t] + LPIS[k] * AIE[k] * (
                SIS_NA[k] * (CE_NA[t] - CI_NA[k][t]) + SIS_K[k] * (
                CE_K[t] - CI_K[k][t]) + SIS_CL[k] * (
                        CE_CL[t] - CI_CL[k][t]) + SIS_HCO3[k] * (
                        CE_HCO3[t] - CI_HCO3[k][t])
                + SIS_H2CO3[k] * (CE_H2CO3[t] - CI_H2CO3[k][t]) + SIS_CO2[k] * (
                        CE_CO2[t] - CI_CO2[k][t]) + SIS_HPO4[k] * (
                        CE_HPO4[t] - CI_HPO4[k][t]) + SIS_H2PO4[k] * (
                        CE_H2PO4[t] - CI_H2PO4[k][t])
                + SIS_UREA[k] * (CE_UREA[t] - CI_UREA[k][t]) + SIS_NH3[k] * (
                        CE_NH3[t] - CI_NH3[k][t]) + SIS_NH4[k] * (
                        CE_NH4[t] - CI_NH4[k][t]) + SIS_HCO2[k] * (
                        CE_HCO2[t] - CI_HCO2[k][t])
                + SIS_H2CO2[k] * (CE_H2CO2[t] - CI_H2CO2) + SIS_GLUC[k] * (
                        CE_GLUC[t] - CI_GLUC[k][t]))

    CME_NA = LMMS(CE_NA[t], CM_NA)
    CME_K = LMMS(CE_K[t], CM_K)
    CME_CL = LMMS(CE_CL[t], CM_CL)
    CME_HCO3 = LMMS(CE_HCO3[t], CM_HCO3)
    CME_H2CO3 = LMMS(CE_H2CO3[t], CM_H2CO3)
    CME_CO2 = LMMS(CE_CO2[t], CM_CO2)
    CME_HPO4 = LMMS(CE_HPO4[t], CM_HPO4)
    CME_H2PO4 = LMMS(CE_H2PO4[t], CM_H2PO4)
    CME_UREA = LMMS(CE_UREA[t], CM_UREA)
    CME_NH3 = LMMS(CE_NH3[t], CM_NH3)
    CME_NH4 = LMMS(CE_NH4[t], CM_NH4)
    CME_H = LMMS(CE_H[t], CM_H)
    CME_HCO2 = LMMS(CE_HCO2[t], CM_HCO2)
    CME_H2CO2 = LMMS(CE_H2CO2[t], CM_H2CO2)
    CME_GLUC = LMMS(CE_GLUC[t], CM_GLUC)

    CES_NA = LMMSC(CE_NA[t], CS_NA)
    CES_K = LMMSC(CE_K[t], CS_K)
    CES_CL = LMMSC(CE_CL[t], CS_CL)
    CES_HCO3 = LMMSC(CE_HCO3[t], CS_HCO3)
    CES_H2CO3 = LMMSC(CE_H2CO3[t], CS_H2CO3)
    CES_CO2 = LMMSC(CE_CO2[t], CS_CO2)
    CES_HPO4 = LMMSC(CE_HPO4[t], CS_HPO4)
    CES_H2PO4 = LMMSC(CE_H2PO4[t], CS_H2PO4)
    CES_UREA = LMMSC(CE_UREA[t], CS_UREA)
    CES_NH3 = LMMSC(CE_NH3[t], CS_NH3)
    CES_NH4 = LMMSC(CE_NH4[t], CS_NH4)
    CES_H = LMMSC(CE_H[t], CS_H)
    CES_HCO2 = LMMSC(CE_HCO2[t], CS_HCO2)
    CES_H2CO2 = LMMSC(CE_H2CO2[t], CS_H2CO2)
    CES_GLUC = LMMSC(CE_GLUC[t], CS_GLUC)

    for k in range(0):
        CMI_NA[k]= LMMSC(CI_NA[k][t], CM_NA)
        CMI_K[k]= LMMSC(CI_K[k][t], CM_K)
        CMI_CL[k]= LMMSC(CI_CL[k][t], CM_CL)
        CMI_HCO3[k]= LMMSC(CI_HCO3[k][t], CM_HCO3)
        CMI_H2CO3[k]= LMMSC(CI_H2CO3[k][t], CM_H2CO3)
        CMI_CO2[k]= LMMSC(CI_CO2[k][t], CM_CO2)
        CMI_HPO4[k]= LMMSC(CI_HPO4[k][t], CM_HPO4)
        CMI_H2PO4[k]= LMMSC(CI_H2PO4[k][t], CM_H2PO4)
        CMI_UREA[k]= LMMSC(CI_UREA[k][t], CM_UREA)
        CMI_NH3[k]= LMMSC(CI_NH3[k][t], CM_NH3)
        CMI_NH4[k]= LMMSC(CI_NH4[k][t], CM_NH4)
        CMI_H[k]= LMMSC(CI_H[k][t], CM_H)
        CMI_HCO2[k]= LMMSC(CI_HCO2[k][t], CM_HCO2)
        CMI_H2CO2[k]= LMMSC(CI_H2CO2[k][t], CM_H2CO2)
        CMI_GLUC[k]= LMMSC(CI_GLUC[k][t], CM_GLUC)

        CIS_NA[k]= LMMSC(CI_NA[k][t], CS_NA)
        CIS_K[k]= LMMSC(CI_K[k][t], CS_K)
        CIS_CL[k]= LMMSC(CI_CL[k][t], CS_CL)
        CIS_HCO3[k]= LMMSC(CI_HCO3[k][t], CS_HCO3)
        CIS_H2CO3[k]= LMMSC(CI_H2CO3[k][t], CS_H2CO3)
        CIS_CO2[k]= LMMSC(CI_CO2[k][t], CS_CO2)
        CIS_HPO4[k]= LMMSC(CI_HPO4[k][t], CS_HPO4)
        CIS_H2PO4[k]= LMMSC(CI_H2PO4[k][t], CS_H2PO4)
        CIS_UREA[k]= LMMSC(CI_UREA[k][t], CS_UREA)
        CIS_NH3[k]= LMMSC(CI_NH3[k][t], CS_NH3)
        CIS_NH4[k]= LMMSC(CI_NH4[k][t], CS_NH4)
        CIS_H[k]= LMMSC(CI_H[k][t], CS_H)
        CIS_HCO2[k]= LMMSC(CI_HCO2[k][t], CS_HCO2)
        CIS_H2CO2[k]= LMMSC(CI_H2CO2[k][t], CS_H2CO2)
        CIS_GLUC[k]= LMMSC(CI_GLUC[k][t], CS_GLUC)

        CIE_NA[k]= LMMSC(CI_NA[k][t], CE_NA[t])
        CIE_K[k]= LMMSC(CI_K[k][t], CE_K[t])
        CIE_CL[k]= LMMSC(CI_CL[k][t], CE_CL[t])
        CIE_HCO3[k]= LMMSC(CI_HCO3[k][t], CE_HCO3[t])
        CIE_H2CO3[k]= LMMSC(CI_H2CO3[k][t], CE_H2CO3[t])
        CIE_CO2[k]= LMMSC(CI_CO2[k][t], CE_CO2[t])
        CIE_HPO4[k]= LMMSC(CI_HPO4[k][t], CE_HPO4[t])
        CIE_H2PO4[k]= LMMSC(CI_H2PO4[k][t], CE_H2PO4[t])
        CIE_UREA[k]= LMMSC(CI_UREA[k][t], CE_UREA[t])
        CIE_NH3[k]= LMMSC(CI_NH3[k][t], CE_NH3[t])
        CIE_NH4[k]= LMMSC(CI_NH4[k][t], CE_NH4[t])
        CIE_H[k]= LMMSC(CI_H[k][t], CE_H[t])
        CIE_HCO2[k]= LMMSC(CI_HCO2[k][t], CE_HCO2[t])
        CIE_H2CO2[k]= LMMSC(CI_H2CO2[k][t], CE_H2CO2[t])
        CIE_GLUC[k]= LMMSC(CI_GLUC[k][t], CE_GLUC[t])

    # CONVECTIVE FLUXES
    FEKM_NA[t] = FEVM[t] * (1.00 - SME_NA) * CME_NA
    FEKM_K[t] = FEVM[t] * (1.00 - SME_K) * CME_K
    FEKM_CL[t] = FEVM[t] * (1.00 - SME_CL) * CME_CL
    FEKM_HCO3[t] = FEVM[t] * (1.00 - SME_HCO3) * CME_HCO3
    FEKM_H2CO3[t] = FEVM[t] * (1.00 - SME_H2CO3) * CME_H2CO3
    FEKM_CO2[t] = FEVM[t] * (1.00 - SME_CO2) * CME_CO2
    FEKM_HPO4[t] = FEVM[t] * (1.00 - SME_HPO4) * CME_HPO4
    FEKM_H2PO4[t] = FEVM[t] * (1.00 - SME_H2PO4) * CME_H2PO4
    FEKM_UREA[t] = FEVM[t] * (1.00 - SME_UREA) * CME_UREA
    FEKM_NH3[t] = FEVM[t] * (1.00 - SME_NH3) * CME_NH3
    FEKM_NH4[t] = FEVM[t] * (1.00 - SME_NH4) * CME_NH4
    FEKM_H[t] = FEVM[t] * (1.00 - SME_H) * CME_H
    FEKM_HCO2[t] = FEVM[t] * (1.00 - SME_HCO2) * CME_HCO2
    FEKM_H2CO2[t] = FEVM[t] * (1.00 - SME_H2CO2) * CME_H2CO2
    FEKM_GLUC[t] = FEVM[t] * (1.00 - SME_GLUC) * CME_GLUC

    FEKS_NA[t] = FEVS[t] * (1.00 - SES_NA) * CES_NA
    FEKS_K[t] = FEVS[t] * (1.00 - SES_K) * CES_K
    FEKS_CL[t] = FEVS[t] * (1.00 - SES_CL) * CES_CL
    FEKS_HCO3[t] = FEVS[t] * (1.00 - SES_HCO3) * CES_HCO3
    FEKS_H2CO3[t] = FEVS[t] * (1.00 - SES_H2CO3) * CES_H2CO3
    FEKS_CO2[t] = FEVS[t] * (1.00 - SES_CO2) * CES_CO2
    FEKS_HPO4[t] = FEVS[t] * (1.00 - SES_HPO4) * CES_HPO4
    FEKS_H2PO4[t] = FEVS[t] * (1.00 - SES_H2PO4) * CES_H2PO4
    FEKS_UREA[t] = FEVS[t] * (1.00 - SES_UREA) * CES_UREA
    FEKS_NH3[t] = FEVS[t] * (1.00 - SES_NH3) * CES_NH3
    FEKS_NH4[t] = FEVS[t] * (1.00 - SES_NH4) * CES_NH4
    FEKS_H[t] = FEVS[t] * (1.00 - SES_H) * CES_H
    FEKS_HCO2[t] = FEVS[t] * (1.00 - SES_HCO2) * CES_HCO2
    FEKS_H2CO2[t] = FEVS[t] * (1.00 - SES_H2CO2) * CES_H2CO2
    FEKS_GLUC[t] = FEVS[t] * (1.00 - SES_GLUC) * CES_GLUC

    for k in range(0):
        FIKM_NA[k][t] = FIVM[k][t] * (1.00 - SMI_NA[k])*CMI_NA[k]
        FIKM_K[k][t] = FIVM[k][t] * (1.00 - SMI_K[k])*CMI_K[k]
        FIKM_CL[k][t] = FIVM[k][t] * (1.00 - SMI_CL[k])*CMI_CL[k]
        FIKM_HCO3[k][t] = FIVM[k][t] * (1.00 - SMI_HCO3[k])*CMI_HCO3[k]
        FIKM_H2CO3[k][t] = FIVM[k][t] * (1.00 - SMI_H2CO3[k])*CMI_H2CO3[k]
        FIKM_CO2[k][t] = FIVM[k][t] * (1.00 - SMI_CO2[k])*CMI_CO2[k]
        FIKM_HPO4[k][t] = FIVM[k][t] * (1.00 - SMI_HPO4[k])*CMI_HPO4[k]
        FIKM_H2PO4[k][t] = FIVM[k][t] * (1.00 - SMI_H2PO4[k])*CMI_H2PO4[k]
        FIKM_UREA[k][t] = FIVM[k][t] * (1.00 - SMI_UREA[k])*CMI_UREA[k]
        FIKM_NH3[k][t] = FIVM[k][t] * (1.00 - SMI_NH3[k])*CMI_NH3[k]
        FIKM_NH4[k][t] = FIVM[k][t] * (1.00 - SMI_NH4[k])*CMI_NH4[k]
        FIKM_H[k][t] = FIVM[k][t] * (1.00 - SMI_H[k])*CMI_H[k]
        FIKM_HCO2[k][t] = FIVM[k][t] * (1.00 - SMI_HCO2[k])*CMI_HCO2[k]
        FIKM_H2CO2[k][t] = FIVM[k][t] * (1.00 - SMI_H2CO2[k])*CMI_H2CO2[k]
        FIKM_GLUC[k][t] = FIVM[k][t] * (1.00 - SMI_GLUC[k])*CMI_GLUC[k]

        FIKS_NA[k][t] = FIVS[k][t] * (1.00 - SIS_NA[k])*CIS_NA[k]
        FIKS_K[k][t] = FIVS[k][t] * (1.00 - SIS_K[k])*CIS_K[k]
        FIKS_CL[k][t] = FIVS[k][t] * (1.00 - SIS_CL[k])*CIS_CL[k]
        FIKS_HCO3[k][t] = FIVS[k][t] * (1.00 - SIS_HCO3[k])*CIS_HCO3[k]
        FIKS_H2CO3[k][t] = FIVS[k][t] * (1.00 - SIS_H2CO3[k])*CIS_H2CO3[k]
        FIKS_CO2[k][t] = FIVS[k][t] * (1.00 - SIS_CO2[k])*CIS_CO2[k]
        FIKS_HPO4[k][t] = FIVS[k][t] * (1.00 - SIS_HPO4[k])*CIS_HPO4[k]
        FIKS_H2PO4[k][t] = FIVS[k][t] * (1.00 - SIS_H2PO4[k])*CIS_H2PO4[k]
        FIKS_UREA[k][t] = FIVS[k][t] * (1.00 - SIS_UREA[k])*CIS_UREA[k]
        FIKS_NH3[k][t] = FIVS[k][t] * (1.00 - SIS_NH3[k])*CIS_NH3[k]
        FIKS_NH4[k][t] = FIVS[k][t] * (1.00 - SIS_NH4[k])*CIS_NH4[k]
        FIKS_H[k][t] = FIVS[k][t] * (1.00 - SIS_H[k])*CIS_H[k]
        FIKS_HCO2[k][t] = FIVS[k][t] * (1.00 - SIS_HCO2[k])*CIS_HCO2[k]
        FIKS_H2CO2[k][t] = FIVS[k][t] * (1.00 - SIS_H2CO2[k])*CIS_H2CO2[k]
        FIKS_GLUC[k][t] = FIVS[k][t] * (1.00 - SIS_GLUC[k])*CIS_GLUC[k]

        JK_NA[k][t] = JV[k][t] * (1.00 - SIS_NA[k])*CIE_NA[k]
        JK_K[k][t] = JV[k][t] * (1.00 - SIS_K[k])*CIE_K[k][t]
        JK_CL[k][t] = JV[k][t] * (1.00 - SIS_CL[k])*CIE_CL[k][t]
        JK_HCO3[k][t] = JV[k][t] * (1.00 - SIS_HCO3[k])*CIE_HCO3[k]
        JK_H2CO3[k][t] = JV[k][t] * (1.00 - SIS_H2CO3[k])*CIE_H2CO3[k]
        JK_CO2[k][t] = JV[k][t] * (1.00 - SIS_CO2[k])*CIE_CO2[k][t]
        JK_HPO4[k][t] = JV[k][t] * (1.00 - SIS_HPO4[k])*CIE_HPO4[k]
        JK_H2PO4[k][t] = JV[k][t] * (1.00 - SIS_H2PO4[k])*CIE_H2PO4[k]
        JK_UREA[k][t] = JV[k][t] * (1.00 - SIS_UREA[k])*CIE_UREA[k]
        JK_NH3[k][t] = JV[k][t] * (1.00 - SIS_NH3[k])*CIE_NH3[k]
        JK_NH4[k][t] = JV[k][t] * (1.00 - SIS_NH4[k])*CIE_NH4[k]
        JK_H[k][t] = JV[k][t] * (1.00 - SIS_H[k])*CIE_H[k]
        JK_HCO2[k][t] = JV[k][t] * (1.00 - SIS_HCO2[k])*CIE_HCO2[k]
        JK_H2CO2[k][t] = JV[k][t] * (1.00 - SIS_H2CO2[k])*CIE_H2CO2[k]
        JK_GLUC[k][t] = JV[k][t] * (1.00 - SIS_GLUC[k])*CIE_GLUC[k]
        print('x: %d t: %d' % (x, t))

    # GOLDMAN  FLUXES
    FEKM_NA[t] = FEKM_NA[t] + GOLDMAN(HME_NA, AME, Z_NA, VM, VE[t], CM_NA,
                                            CE_NA[t], 1)
    FEKM_K[t] = FEKM_K[t] + GOLDMAN(HME_K, AME, Z_K, VM, VE[t], CM_K, CE_K[t], 1)
    FEKM_CL[t] = FEKM_CL[t] + GOLDMAN(HME_CL, AME, Z_CL, VM, VE[t], CM_CL, CE_CL[t], 1)
    FEKM_HCO3[t] = FEKM_HCO3[t] + GOLDMAN(HME_HCO3, AME, Z_HCO3, VM, VE[t], CM_HCO3, CE_HCO3[t],
                                                1)
    FEKM_H2CO3[t] = FEKM_H2CO3[t] + GOLDMAN(HME_H2CO3, AME, Z_H2CO3, VM, VE[t], CM_H2CO3,
                                                  CE_H2CO3[t], 1)
    FEKM_CO2[t] = FEKM_CO2[t] + GOLDMAN(HME_CO2, AME, Z_CO2, VM, VE[t], CM_CO2, CE_CO2[t], 1)
    FEKM_HPO4[t] = FEKM_HPO4[t] + GOLDMAN(HME_HPO4, AME, Z_HPO4, VM, VE[t], CM_HPO4,
                                                CE_HPO4[t], 1)
    FEKM_H2PO4[t] = FEKM_H2PO4[t] + GOLDMAN(HME_H2PO4, AME, Z_H2PO4, VM, VE[t], CM_H2PO4,
                                                  CE_H2PO4[t], 1)
    FEKM_UREA[t] = FEKM_UREA[t] + GOLDMAN(HME_UREA, AME, Z_UREA, VM, VE[t], CM_UREA, CE_UREA[t],
                                                1)
    FEKM_NH3[t] = FEKM_NH3[t] + GOLDMAN(HME_NH3, AME, Z_NH3, VM, VE[t], CM_NH3, CE_NH3[t], 1)
    FEKM_NH4[t] = FEKM_NH4[t] + GOLDMAN(HME_NH4, AME, Z_NH4, VM, VE[t], CM_NH4, CE_NH4[t], 1)
    FEKM_H[t] = FEKM_H[t] + GOLDMAN(HME_H, AME, Z_H, VM, VE[t], CM_H, CE_H[t], 1)
    FEKM_HCO2[t] = FEKM_HCO2[t] + GOLDMAN(HME_HCO2, AME, Z_HCO2, VM, VE[t], CM_HCO2, CE_HCO2[t],
                                                1)
    FEKM_H2CO2[t] = FEKM_H2CO2[t] + GOLDMAN(HME_H2CO2, AME, Z_H2CO2, VM, VE[t], CM_H2CO2,
                                                  CE_H2CO2[t], 1)
    FEKM_GLUC[t] = FEKM_GLUC[t] + GOLDMAN(HME_GLUC, AME, Z_GLUC, VM, VE[t], CM_GLUC, CE_GLUC[t],
                                                1)

    FEKS_NA[t] = FEKS_NA[t] + GOLDMAN(HES_NA, AE[t], Z_NA, VE[t], VS, CE_NA[t],
                                            CS_NA, 1)
    FEKS_K[t] = FEKS_K[t] + GOLDMAN(HES_K, AE[t], Z_K, VE[t], VS, CE_K[t], CS_K, 1)
    FEKS_CL[t] = FEKS_CL[t] + GOLDMAN(HES_CL, AE[t], Z_CL, VE[t], VS, CE_CL[t], CS_CL, 1)
    FEKS_HCO3[t] = FEKS_HCO3[t] + GOLDMAN(HES_HCO3, AE[t], Z_HCO3, VE[t], VS, CE_HCO3[t],
                                                CS_HCO3, 1)
    FEKS_H2CO3[t] = FEKS_H2CO3[t] + GOLDMAN(HES_H2CO3, AE[t], Z_H2CO3, VE[t], VS, CE_H2CO3[t],
                                                  CS_H2CO3, 1)
    FEKS_CO2[t] = FEKS_CO2[t] + GOLDMAN(HES_CO2, AE[t], Z_CO2, VE[t], VS, CE_CO2[t], CS_CO2[t], 1)
    FEKS_HPO4[t] = FEKS_HPO4[t] + GOLDMAN(HES_HPO4, AE[t], Z_HPO4, VE[t], VS, CE_HPO4[t],
                                                CS_HPO4, 1)
    FEKS_H2PO4[t] = FEKS_H2PO4[t] + GOLDMAN(HES_H2PO4, AE[t], Z_H2PO4, VE[t], VS, CE_H2PO4[t],
                                                  CS_H2PO4, 1)
    FEKS_UREA[t] = FEKS_UREA[t] + GOLDMAN(HES_UREA, AE[t], Z_UREA, VE[t], VS, CE_UREA[t],
                                                CS_UREA, 1)
    FEKS_NH3[t] = FEKS_NH3[t] + GOLDMAN(HES_NH3, AE[t], Z_NH3, VE[t], VS, CE_NH3[t], CS_NH3, 1)
    FEKS_NH4[t] = FEKS_NH4[t] + GOLDMAN(HES_NH4, AE[t], Z_NH4, VE[t], VS, CE_NH4[t], CS_NH4, 1)
    FEKS_H[t] = FEKS_H[t] + GOLDMAN(HES_H, AE[t], Z_H, VE[t], VS, CE_H[t], CS_H, 1)
    FEKS_HCO2[t] = FEKS_HCO2[t] + GOLDMAN(HES_HCO2, AE[t], Z_HCO2, VE[t], VS, CE_HCO2[t],
                                                CS_HCO2, 1)
    for k in range(0):

        FIKM_NA[k][t] = FIKM_NA[k][t] + GOLDMAN(HMI_NA, AMI[k], Z_NA, VM, VI[k][t], CM_NA,
                                                CI_NA[k][t], 1)
        FIKM_K[k][t] = FIKM_K[k][t] + GOLDMAN(HMI_K, AMI[k], Z_K, VM, VI[k][t], CM_K, CI_K[k][t], 1)
        FIKM_CL[k][t] = FIKM_CL[k][t] + GOLDMAN(HMI_CL, AMI[k], Z_CL, VM, VI[k][t], CM_CL, CI_CL[k][t], 1)
        FIKM_HCO3[k][t] = FIKM_HCO3[k][t] + GOLDMAN(HMI_HCO3, AMI[k], Z_HCO3, VM, VI[k][t], CM_HCO3, CI_HCO3[k][t], 1)
        FIKM_H2CO3[k][t] = FIKM_H2CO3[k][t] + GOLDMAN(HMI_H2CO3, AMI[k], Z_H2CO3, VM, VI[k][t], CM_H2CO3,
                                                      CI_H2CO3[k][t], 1)
        FIKM_CO2[k][t] = FIKM_CO2[k][t] + GOLDMAN(HMI_CO2, AMI[k], Z_CO2, VM, VI[k][t], CM_CO2, CI_CO2[k][t], 1)
        FIKM_HPO4[k][t] = FIKM_HPO4[k][t] + GOLDMAN(HMI_HPO4, AMI[k], Z_HPO4, VM, VI[k][t], CM_HPO4,
                                                    CI_HPO4[k][t], 1)
        FIKM_H2PO4[k][t] = FIKM_H2PO4[k][t] + GOLDMAN(HMI_H2PO4, AMI[k], Z_H2PO4, VM, VI[k][t], CM_H2PO4,
                                                      CI_H2PO4[k][t], 1)
        FIKM_UREA[k][t] = FIKM_UREA[k][t] + GOLDMAN(HMI_UREA, AMI[k], Z_UREA, VM, VI[k][t], CM_UREA, CI_UREA[k][t], 1)
        FIKM_NH3[k][t] = FIKM_NH3[k][t] + GOLDMAN(HMI_NH3, AMI[k], Z_NH3, VM, VI[k][t], CM_NH3, CI_NH3[k][t], 1)
        FIKM_NH4[k][t] = FIKM_NH4[k][t] + GOLDMAN(HMI_NH4, AMI[k], Z_NH4, VM, VI[k][t], CM_NH4, CI_NH4[k][t], 1)
        FIKM_H[k][t] = FIKM_H[k][t] + GOLDMAN(HMI_H, AMI[k], Z_H, VM, VI[k][t], CM_H, CI_H[k][t], 1)
        FIKM_HCO2[k][t] = FIKM_HCO2[k][t] + GOLDMAN(HMI_HCO2, AMI[k], Z_HCO2, VM, VI[k][t], CM_HCO2, CI_HCO2[k][t], 1)
        FIKM_H2CO2[k][t] = FIKM_H2CO2[k][t] + GOLDMAN(HMI_H2CO2, AMI[k], Z_H2CO2, VM, VI[k][t], CM_H2CO2,
                                                      CI_H2CO2[k][t], 1)
        FIKM_GLUC[k][t] = FIKM_GLUC[k][t] + GOLDMAN(HMI_GLUC, AMI[k], Z_GLUC, VM, VI[k][t], CM_GLUC, CI_GLUC[k][t], 1)

        JK_NA[k][t] = JK_NA[k][t] + GOLDMAN(HIS_NA, AIE[k], Z_NA, VI[k][t], VE[t], CI_NA[k][t],
                                            CE_NA[t], 1)
        JK_K[k][t] = JK_K[k][t] + GOLDMAN(HIS_K, AIE[k], Z_K, VI[k][t], VE[t], CI_K[k][t], CE_K[t], 1)
        JK_CL[k][t] = JK_CL[k][t] + GOLDMAN(HIS_CL, AIE[k], Z_CL, VI[k][t], VE[t], CI_CL[k][t], CE_CL[t], 1)
        JK_HCO3[k][t] = JK_HCO3[k][t] + GOLDMAN(HIS_HCO3, AIE[k], Z_HCO3, VI[k][t], VE[t], CI_HCO3[k][t], CE_HCO3[t], 1)
        JK_H2CO3[k][t] = JK_H2CO3[k][t] + GOLDMAN(HIS_H2CO3, AIE[k], Z_H2CO3, VI[k][t], VE[t], CI_H2CO3[k][t],
                                                  CE_H2CO3[t], 1)
        JK_CO2[k][t] = JK_CO2[k][t] + GOLDMAN(HIS_CO2, AIE[k], Z_CO2, VI[k][t], VE[t], CI_CO2[k][t], CE_CO2[t], 1)
        JK_HPO4[k][t] = JK_HPO4[k][t] + GOLDMAN(HIS_HPO4, AIE[k], Z_HPO4, VI[k][t], VE[t], CI_HPO4[k][t],
                                                CE_HPO4[t], 1)
        JK_H2PO4[k][t] = JK_H2PO4[k][t] + GOLDMAN(HIS_H2PO4, AIE[k], Z_H2PO4, VI[k][t], VE[t], CI_H2PO4[k][t],
                                                  CE_H2PO4[t], 1)
        JK_UREA[k][t] = JK_UREA[k][t] + GOLDMAN(HIS_UREA, AIE[k], Z_UREA, VI[k][t], VE[t], CI_UREA[k][t], CE_UREA[t], 1)
        JK_NH3[k][t] = JK_NH3[k][t] + GOLDMAN(HIS_NH3, AIE[k], Z_NH3, VI[k][t], VE[t], CI_NH3[k][t], CE_NH3[t], 1)
        JK_NH4[k][t] = JK_NH4[k][t] + GOLDMAN(HIS_NH4, AIE[k], Z_NH4, VI[k][t], VE[t], CI_NH4[k][t], CE_NH4[t], 1)
        JK_H[k][t] = JK_H[k][t] + GOLDMAN(HIS_H, AIE[k], Z_H, VI[k][t], VE[t], CI_H[k][t], CE_H[t], 1)
        JK_HCO2[k][t] = JK_HCO2[k][t] + GOLDMAN(HIS_HCO2, AIE[k], Z_HCO2, VI[k][t], VE[t], CI_HCO2[k][t], CE_HCO2[t], 1)
        JK_H2CO2[k][t] = JK_H2CO2[k][t] + GOLDMAN(HIS_H2CO2, AIE[k], Z_H2CO2, VI[k][t], VE[t], CI_H2CO2[k][t],
                                                  CE_H2CO2[t], 1)
        JK_GLUC[k][t] = JK_GLUC[k][t] + GOLDMAN(HIS_GLUC, AIE[k], Z_GLUC, VI[k][t], VE[t], CI_GLUC[k][t], CE_GLUC[t], 1)


        FIKS_NA[k][t] = FIKS_NA[k][t] + GOLDMAN(HIS_NA, AIS[k], Z_NA, VI[k][t], VS, CI_NA[k][t],
                                                CS_NA, 1)
        FIKS_K[k][t] = FIKS_K[k][t] + GOLDMAN(HIS_K, AIS[k], Z_K, VI[k][t], VS, CI_K[k][t], CS_K, 1)
        FIKS_CL[k][t] = FIKS_CL[k][t] + GOLDMAN(HIS_CL, AIS[k], Z_CL, VI[k][t], VS, CI_CL[k][t], CS_CL, 1)
        FIKS_HCO3[k][t] = FIKS_HCO3[k][t] + GOLDMAN(HIS_HCO3, AIS[k], Z_HCO3, VI[k][t], VS, CI_HCO3[k][t], CS_HCO3, 1)
        FIKS_H2CO3[k][t] = FIKS_H2CO3[k][t] + GOLDMAN(HIS_H2CO3, AIS[k], Z_H2CO3, VI[k][t], VS, CI_H2CO3[k][t],
                                                      CS_H2CO3, 1)
        FIKS_CO2[k][t] = FIKS_CO2[k][t] + GOLDMAN(HIS_CO2, AIS[k], Z_CO2, VI[k][t], VS, CI_CO2[k][t], CS_CO2, 1)
        FIKS_HPO4[k][t] = FIKS_HPO4[k][t] + GOLDMAN(HIS_HPO4, AIS[k], Z_HPO4, VI[k][t], VS, CI_HPO4[k][t],
                                                    CS_HPO4, 1)
        FIKS_H2PO4[k][t] = FIKS_H2PO4[k][t] + GOLDMAN(HIS_H2PO4, AIS[k], Z_H2PO4, VI[k][t], VS, CI_H2PO4[k][t],
                                                      CS_H2PO4, 1)
        FIKS_UREA[k][t] = FIKS_UREA[k][t] + GOLDMAN(HIS_UREA, AIS[k], Z_UREA, VI[k][t], VS, CI_UREA[k][t], CS_UREA, 1)
        FIKS_NH3[k][t] = FIKS_NH3[k][t] + GOLDMAN(HIS_NH3, AIS[k], Z_NH3, VI[k][t], VS, CI_NH3[k][t], CS_NH3, 1)
        FIKS_NH4[k][t] = FIKS_NH4[k][t] + GOLDMAN(HIS_NH4, AIS[k], Z_NH4, VI[k][t], VS, CI_NH4[k][t], CS_NH4, 1)
        FIKS_H[k][t] = FIKS_H[k][t] + GOLDMAN(HIS_H, AIS[k], Z_H, VI[k][t], VS, CI_H[k][t], CS_H, 1)
        FIKS_HCO2[k][t] = FIKS_HCO2[k][t] + GOLDMAN(HIS_HCO2, AIS[k], Z_HCO2, VI[k][t], VS, CI_HCO2[k][t], CS_HCO2, 1)
        FIKS_H2CO2[k][t] = FIKS_H2CO2[k][t] + GOLDMAN(HIS_H2CO2, AIS[k], Z_H2CO2, VI[k][t], VS, CI_H2CO2[k][t],
                                                      CS_H2CO2, 1)
        FIKS_GLUC[k][t] = FIKS_GLUC[k][t] + GOLDMAN(HIS_GLUC, AIS[k], Z_GLUC, VI[k][t], VS, CI_GLUC[k][t], CS_GLUC, 1)

        # Net Cotransporters

    for k in range(0):
        SGLT = SGLT_MI(CM_NA, CI_NA[k][t], CM_GLUC, CI_GLUC[k][t], Z_NA, Z_GLUC, VM, VI[k][t], AMI[k], LMI_NAGLUC,
                       1)
        NA_MI_NAGLUC[k] = SGLT[0]
        GLUC_MI_NAGLUC[k] = SGLT[1]
        NAH2PO4 = NAH2PO4_MI(CM_NA, CI_NA[k][t], CM_H2PO4, CI_H2PO4[k][t], Z_NA, Z_H2PO4, VM, VI[k][t], AMI[k],
                             LMI_NAH2PO4, 1)
        NA_MI_NAH2PO4[k] = NAH2PO4[0]
        H2PO4_MI_NAH2PO4[k] = NAH2PO4[1]
        CLHCO3 = CLHCO3_MI(CM_CL, CI_CL[k][t], CM_HCO3, CI_HCO3[k][t], Z_CL, Z_HCO3, VM, VI[k][t], AMI[k],
                           LMI_CLHCO3, 1)
        CL_MI_CLHCO3[k] = CLHCO3[0]
        HCO3_MI_CLHCO3[k] = CLHCO3[1]
        CLHCO2 = CLHCO2_MI(CM_CL, CI_CL[k][t], CM_HCO2, CI_HCO2[k][t], Z_CL, Z_HCO2, VM, VI[k][t], AMI[k],
                           LMI_CLHCO2, 1)
        CL_MI_CLHCO2[k] = CLHCO2[0]
        HCO2_MI_CLHCO2[k] = CLHCO2[1]
        NAHCO3 = NAHCO3_IS(CI_NA[k][t], CS_NA, CI_HCO3[k][t], CS_HCO3, Z_NA, Z_HCO3, VI[k][t], VS, AIS[k],
                           LIS_NAHCO3, 1)
        NA_IS_NAHCO3[k] = NAHCO3[0]
        HCO3_IS_NAHCO3[k] = NAHCO3[1]
        KCL = KCL_IS(CI_K[k][t], CS_K, CI_CL[k][t], CS_CL, Z_K, Z_CL, VI[k][t], VS, AIS[k], LIS_KCL, 1)
        K_IS_KCL[k] = KCL[0]
        CL_IS_KCL[k] = KCL[1]
        NA_CLHCO3 = NA_CLHCO3_IS(CI_NA[k][t], CS_NA, CI_CL[k][t], CS_CL, CI_HCO3[k][t], CS_HCO3, Z_NA, Z_CL, Z_HCO3,
                                 VI[k][t], VS, AIS[k], LIS_NA_CLHCO3, 1)
        NA_IS_NA_CLHCO3[k] = NA_CLHCO3[0]
        CL_IS_NA_CLHCO3[k] = NA_CLHCO3[1]
        HCO3_IS_NA_CLHCO3[k] = NA_CLHCO3[2]
        # THE NAH EXCHANGER TRANSLATE CONCENTRATIONS TO THE NAH MODEL
        MYNAH = NAH(CI_H[k][t], CI_NA[k][t], CI_NH4[k][t], CM_H, CM_NA, CM_NH4, 1)
        JNAH_NA = MYNAH[0]
        JNAH_H = MYNAH[1]
        JNAH_NH4 = MYNAH[2]
        JNHE3_NA[k] = NNHE3 * AMI[k] * JNAH_NA
        JNHE3_H[k] = NNHE3 * AMI[k] * JNAH_H
        JNHE3_NH4[k] = NNHE3 * AMI[k] * JNAH_NH4

        FIKM_NA[k][t] = FIKM_NA[k][t] + NA_MI_NAGLUC[k] + NA_MI_NAH2PO4[k] + JNHE3_NA[k]
        FIKM_CL[k][t] = FIKM_CL[k][t] + CL_MI_CLHCO2[k] + CL_MI_CLHCO3[k]
        FIKM_HCO3[k][t] = FIKM_HCO3[k][t] + HCO3_MI_CLHCO3[k]
        FIKM_H2PO4[k][t] = FIKM_H2PO4[k][t] + H2PO4_MI_NAH2PO4[k]
        FIKM_HCO2[k][t] = FIKM_HCO2[k][t] + HCO2_MI_CLHCO2[k]
        FIKM_GLUC[k][t] = FIKM_GLUC[k][t] + GLUC_MI_NAGLUC[k]
        FIKM_H[k][t] = FIKM_H[k][t] + JNHE3_H[k]
        FIKM_NH4[k][t] = FIKM_NH4[k][t] + JNHE3_NH4[k]

        FIKS_NA[k][t] = FIKS_NA[k][t] + NA_IS_NAHCO3[k] + NA_IS_NA_CLHCO3[k]
        FIKS_K[k][t] = FIKS_K[k][t] + K_IS_KCL[k]
        FIKS_CL[k][t] = FIKS_CL[k][t] + CL_IS_KCL[k] + CL_IS_NA_CLHCO3[k]
        FIKS_HCO3[k][t] = FIKS_HCO3[k][t] + HCO3_IS_NA_CLHCO3[k] + HCO3_IS_NAHCO3[k]

        JK_NA[k][t] = JK_NA[k][t] + NA_IS_NAHCO3[k] + NA_IS_NA_CLHCO3[k]
        JK_K[k][t] = JK_K[k][t] + K_IS_KCL[k]
        JK_CL[k][t] = JK_CL[k][t] + CL_IS_KCL[k] + CL_IS_NA_CLHCO3[k]
        JK_HCO3[k][t] = JK_HCO3[k][t] + HCO3_IS_NA_CLHCO3[k] + HCO3_IS_NAHCO3[k]

    for k in range(0):
        # SODIUM PUMPS
        NAK = NAK_ATP(CI_K[k][t], CS_K, CI_NA[k][t], CS_NA, CE_K[k][t], CE_NH4[k][t], 1)
        ATIS_NA[k] = NAK[0]
        ATIS_K[k] = NAK[1]
        ATIS_NH4[k] = NAK[2]
        ATMI_H = AT_MI_H(CM_H, CI_H[k][t], VM, VI[k][t], Z_H, 1)

        JK_NA[k][t] = JK_NA[k][t] + AIE[k] * ATIS_NA[k]
        JK_K[k][t] = JK_K[k][t] + AIE[k] * ATIS_K[k]
        JK_NH4[k][t] = JK_NH4[k][t] + AIE[k] * ATIS_NH4[k]
        FIKS_NA[k][t] = FIKS_NA[k][t] + AIS[k] * ATIS_NA[k]
        FIKS_K[k][t] = FIKS_K[k][t] + AIS[k] * ATIS_K[k]
        FIKS_NH4[k][t] = FIKS_NH4[k][t] + AIS[k] * ATIS_NH4[k]

        JNAK_NA[k]= AIE[k]*ATIS_NA[k]+AIS[k]*ATIS_NA[k]
        JNAK_K[k]= AIE[k]*ATIS_K[k]+AIS[k]*ATIS_K[k]
        JNAK_NH4[k]= AIE[k]*ATIS_NH4[k]+AIS[k]*ATIS_NH4[k]

        #PROTON PUMPS
        FIKM_H[k][t] = FIKM_H[k][t] + AMI[k] * ATMI_H
#ESTABLISH THE ERROR VECTORS, THE "PHI" ARRAY.

    #FIRST FOR THE INTERSPACE  ELECTRONEUTRALITY
    PHIE_EN = 0
    PHIE_EN = PHIE_EN + Z_NA * CE_NA[t] + Z_K * CE_K[t] + Z_CL * CE_CL[
        t] + Z_HCO3 * CE_HCO3[t] + Z_H2CO3 * CE_H2CO3[t] \
                 + Z_CO2 * CE_CO2[t] + Z_HPO4 * CE_HPO4[t] + Z_H2PO4 * CE_H2PO4[
                     t] + Z_UREA * CE_UREA[t] + Z_NH3 * CE_NH3[t] \
                 + Z_NH4 * CE_NH4[t] + Z_H * CE_H[t] + Z_HCO2 * CE_HCO2[t] + Z_H2CO2 * CE_H2CO2[t] + Z_GLUC * CE_GLUC[t]
    if t==1:
    # MASS CONSERVATION IN THE TIME-DEPENDENT CASE
        PHIE_VLM = FEVS[t] - FEVM[t] + RTAU * (CHVL[t] - CHVL[t - 1])

        QE_NA = FEKS_NA[t] - FEKM_NA[t] + RTAU * (CE_NA[t] * CHVL[t] - CE_NA[t - 1] * CHVL[t - 1])
        QE_K = FEKS_K[t] - FEKM_K[t] + RTAU * (CE_K[t] * CHVL[t] - CE_K[t - 1] * CHVL[t - 1])
        QE_CL = FEKS_CL[t] - FEKM_CL[t] + RTAU * (CE_CL[t] * CHVL[t] - CE_CL[t - 1] * CHVL[t - 1])
        QE_HCO3 = FEKS_HCO3[t] - FEKM_HCO3[t] + RTAU * (CE_HCO3[t] * CHVL[t] - CE_HCO3[t - 1] * CHVL[t - 1])
        QE_H2CO3 = FEKS_H2CO3[t] - FEKM_H2CO3[t] + RTAU * (CE_H2CO3[t] * CHVL[t] - CE_H2CO3[t - 1] * CHVL[t - 1])
        QE_CO2 = FEKS_CO2[t] - FEKM_CO2[t] + RTAU * (CE_CO2[t] * CHVL[t] - CE_CO2[t - 1] * CHVL[t - 1])
        QE_HPO4 = FEKS_HPO4[t] - FEKM_HPO4[t] + RTAU * (CE_HPO4[t] * CHVL[t] - CE_HPO4[t - 1] * CHVL[t - 1])
        QE_H2PO4 = FEKS_H2PO4[t] - FEKM_H2PO4[t] + RTAU * (CE_H2PO4[t] * CHVL[t] - CE_H2PO4[t - 1] * CHVL[t - 1])
        QE_UREA = FEKS_UREA[t] - FEKM_UREA[t] + RTAU * (CE_UREA[t] * CHVL[t] - CE_UREA[t - 1] * CHVL[t - 1])
        QE_NH3 = FEKS_NH3[t] - FEKM_NH3[t] + RTAU * (CE_NH3[t] * CHVL[t] - CE_NH3[t - 1] * CHVL[t - 1])
        QE_NH4 = FEKS_NH4[t] - FEKM_NH4[t] + RTAU * (CE_NH4[t] * CHVL[t] - CE_NH4[t - 1] * CHVL[t - 1])
        QE_H = FEKS_H[t] - FEKM_H[t] + RTAU * (CE_H[t] * CHVL[t] - CE_H[t - 1] * CHVL[t - 1])
        QE_HCO2 = FEKS_HCO2[t] - FEKM_HCO2[t] + RTAU * (CE_HCO2[t] * CHVL[t] - CE_HCO2[t - 1] * CHVL[t - 1])
        QE_H2CO2 = FEKS_H2CO2[t] - FEKM_H2CO2[t] + RTAU * (CE_H2CO2[t] * CHVL[t] - CE_H2CO2[t - 1] * CHVL[t - 1])
        QE_GLUC = FEKS_GLUC[t] - FEKM_GLUC[t] + RTAU * (CE_GLUC[t] * CHVL[t] - CE_GLUC[t - 1] * CHVL[t - 1])

        for k in range(0):
            PHIE_VLM = PHIE_VLM - JV[k][t]
            QE_NA = QE_NA - JK_NA[k][t]
            QE_K = QE_K - JK_K[k][t]
            QE_CL = QE_CL - JK_CL[k][t]
            QE_HCO3 = QE_HCO3 - JK_HCO3[k][t]
            QE_H2CO3 = QE_H2CO3 - JK_H2CO3[k][t]
            QE_CO2 = QE_CO2 - JK_CO2[k][t]
            QE_HPO4 = QE_HPO4 - JK_HPO4[k][t]
            QE_H2PO4 = QE_H2PO4 - JK_H2PO4[k][t]
            QE_UREA = QE_UREA - JK_UREA[k][t]
            QE_NH3 = QE_NH3 - JK_NH3[k][t]
            QE_NH4 = QE_NH4 - JK_NH4[k][t]
            QE_H = QE_H - JK_H[k][t]
            QE_HCO2 = QE_HCO2 - JK_HCO2[k][t]
            QE_H2CO2 = QE_H2CO2 - JK_H2CO2[k][t]
            QE_GLUC = QE_GLUC - JK_GLUC[k][t]

    else:
 #  MASS  CONSERVATION IN THE STEADY - STATE CASE
        PHIE_VLM = FEVS[t] - FEVM[t]
        QE_NA = FEKS_NA[t] - FEKM_NA[t]
        QE_K = FEKS_K[t] - FEKM_K[t]
        QE_CL = FEKS_CL[t] - FEKM_CL[t]
        QE_HCO3 = FEKS_HCO3[t] - FEKM_HCO3[t]
        QE_H2CO3 = FEKS_H2CO3[t] - FEKM_H2CO3[t]
        QE_CO2 = FEKS_CO2[t] - FEKM_CO2[t]
        QE_HPO4 = FEKS_HPO4[t] - FEKM_HPO4[t]
        QE_H2PO4 = FEKS_H2PO4[t] - FEKM_H2PO4[t]
        QE_UREA = FEKS_UREA[t] - FEKM_UREA[t]
        QE_NH3 = FEKS_NH3[t] - FEKM_NH3[t]
        QE_NH4 = FEKS_NH4[t] - FEKM_NH4[t]
        QE_H = FEKS_H[t] - FEKM_H[t]
        QE_HCO2 = FEKS_HCO2[t] - FEKM_HCO2[t]
        QE_H2CO2 = FEKS_H2CO2[t] - FEKM_H2CO2[t]
        QE_GLUC = FEKS_GLUC[t] - FEKM_GLUC[t]

        for k in range(0):
            PHIE_VLM = PHIE_VLM - JV[k][t]
            QE_NA = QE_NA - JK_NA[k][t]
            QE_K = QE_K - JK_K[k][t]
            QE_CL = QE_CL - JK_CL[k][t]
            QE_HCO3 = QE_HCO3 - JK_HCO3[k][t]
            QE_H2CO3 = QE_H2CO3 - JK_H2CO3[k][t]
            QE_CO2 = QE_CO2 - JK_CO2[k][t]
            QE_HPO4 = QE_HPO4 - JK_HPO4[k][t]
            QE_H2PO4 = QE_H2PO4 - JK_H2PO4[k][t]
            QE_UREA = QE_UREA - JK_UREA[k][t]
            QE_NH3 = QE_NH3 - JK_NH3[k][t]
            QE_NH4 = QE_NH4 - JK_NH4[k][t]
            QE_H = QE_H - JK_H[k][t]
            QE_HCO2 = QE_HCO2 - JK_HCO2[k][t]
            QE_H2CO2 = QE_H2CO2 - JK_H2CO2[k][t]
            QE_GLUC = QE_GLUC - JK_GLUC[k][t]

        return



#THEN FOR THE CELLS  ELECTRONEUTRALITY
    for k in range(0):

       PHII_EN = IMP[k]* ZIMP - CBUF[k][t]
       PHII_EN = PHII_EN + Z_NA * CI_NA[k][t] + Z_K * CI_K[k][t] \
                    + Z_CL * CI_CL[k][t] + Z_HCO3 * CI_HCO3[k][t] + Z_H2CO3 * CI_H2CO3[k][
                        t] + Z_CO2 * CI_CO2[k][t] \
                    + Z_HPO4 * CI_HPO4[k][t] + Z_H2PO4 * CI_H2PO4[k][t] + Z_UREA * CI_UREA[k][t] \
                    + Z_NH3 * CI_NH3[k][t] + Z_NH4 * CI_NH4[k][t] + Z_H * CI_H[k][t] \
                    + Z_HCO2 * CI_HCO2[k][t] + Z_H2CO2 * CI_H2CO2[k][t] + Z_GLUC * CI_GLUC[k][t]

       if t==1:
            #MASS CONSERVATION IN THE TIME - DEPENDENT CASE
            PHII_VLM[k] = FIVS[k][t] - FIVM[k][t] + JV[k][t] + RTAU * (CLVL[k][t] - CLVL[k][t - 1])

            QI_NA[k] = FIKS_NA[k][t] - FIKM_NA[k][t] + JK_NA[k][t] + RTAU * (
                        CI_NA[k][t] * CLVL[k][t] - CI_NA[k][t - 1] * CLVL[k][t - 1])
            QI_K[k] = FIKS_K[k][t] - FIKM_K[k][t] + JK_K[k][t] + RTAU * (
                        CI_K[k][t] * CLVL[k][t] - CI_K[k][t - 1] * CLVL[k][t - 1])
            QI_CL[k] = FIKS_CL[k][t] - FIKM_CL[k][t] + JK_CL[k][t] + RTAU * (
                        CI_CL[k][t] * CLVL[k][t] - CI_CL[k][t - 1] * CLVL[k][t - 1])
            QI_HCO3[k] = FIKS_HCO3[k][t] - FIKM_HCO3[k][t] + JK_HCO3[k][t] + RTAU * (
                        CI_HCO3[k][t] * CLVL[k][t] - CI_HCO3[k][t - 1] * CLVL[k][t - 1])
            QI_H2CO3[k] = FIKS_H2CO3[k][t] - FIKM_H2CO3[k][t] + JK_H2CO3[k][t] + RTAU * (
                        CI_H2CO3[k][t] * CLVL[k][t] - CI_H2CO3[k][t - 1] * CLVL[k][t - 1])
            QI_CO2[k] = FIKS_CO2[k][t] - FIKM_CO2[k][t] + JK_CO2[k][t] + RTAU * (
                        CI_CO2[k][t] * CLVL[k][t] - CI_CO2[k][t - 1] * CLVL[k][t - 1])
            QI_HPO4[k] = FIKS_HPO4[k][t] - FIKM_HPO4[k][t] + JK_HPO4[k][t] + RTAU * (
                        CI_HPO4[k][t] * CLVL[k][t] - CI_HPO4[k][t - 1] * CLVL[k][t - 1])
            QI_H2PO4[k] = FIKS_H2PO4[k][t] - FIKM_H2PO4[k][t] + JK_H2PO4[k][t] + RTAU * (
                        CI_HPO4[k][t] * CLVL[k][t] - CI_HPO4[k][t - 1] * CLVL[k][t - 1])
            QI_UREA[k] = FIKS_UREA[k][t] - FIKM_UREA[k][t] + JK_UREA[k][t] + RTAU * (
                        CI_UREA[k][t] * CLVL[k][t] - CI_UREA[k][t - 1] * CLVL[k][t - 1])
            QI_NH3[k] = FIKS_NH3[k][t] - FIKM_NH3[k][t] + JK_NH3[k][t] + RTAU * (
                        CI_NH3[k][t] * CLVL[k][t] - CI_NH3[k][t - 1] * CLVL[k][t - 1])
            QI_NH4[k] = FIKS_NH4[k][t] - FIKM_NH4[k][t] + JK_NH4[k][t] + RTAU * (
                        CI_NH4[k][t] * CLVL[k][t] - CI_NH4[k][t - 1] * CLVL[k][t - 1])
            QI_H[k] = FIKS_H[k][t] - FIKM_H[k][t] + JK_H[k][t] + RTAU * (
                        CI_H[k][t] * CLVL[k][t] - CI_H[k][t - 1] * CLVL[k][t - 1])
            QI_HCO2[k] = FIKS_HCO2[k][t] - FIKM_HCO2[k][t] + JK_HCO2[k][t] + RTAU * (
                        CI_HCO2[k][t] * CLVL[k][t] - CI_HCO2[k][t - 1] * CLVL[k][t - 1])
            QI_H2CO2[k] = FIKS_H2CO2[k][t] - FIKM_H2CO2[k][t] + JK_H2CO2[k][t] + RTAU * (
                        CI_H2CO2[k][t] * CLVL[k][t] - CI_H2CO2[k][t - 1] * CLVL[k][t - 1])
            QI_GLUC[k] = FIKS_GLUC[k][t] - FIKM_GLUC[k][t] + JK_GLUC[k][t] + RTAU * (
                        CI_GLUC[k][t] * CLVL[k][t] - CI_GLUC[k][t - 1] * CLVL[k][t - 1])

            # THE PROTON FLUX MUST INCLUDE THE CELLULAR BUFFERS
            QI_H[k] =   QI_H[k] - RTAU * (
                    CBUF[k][t] * CLVL[k][t] - CBUF[k][t-1] * CLVL[k][t - 1])
       else:
            ## MASS CONSERVATION IN THE STEADY-STATE CASE
            PHII_VLM[k] = FIVS[k][t] - FIVM[k][t] + JV[k][t]
            QI_NA[k] = FIKS_NA[k][t] - FIKM_NA[k][t] + JK_NA[k][t]
            QI_K[k] = FIKS_K[k][t] - FIKM_K[k][t] + JK_K[k][t]
            QI_CL[k] = FIKS_CL[k][t] - FIKM_CL[k][t] + JK_CL[k][t]
            QI_HCO3[k] = FIKS_HCO3[k][t] - FIKM_HCO3[k][t] + JK_HCO3[k][t]
            QI_H2CO3[k] = FIKS_H2CO3[k][t] - FIKM_H2CO3[k][t] + JK_H2CO3[k][t]
            QI_CO2[k] = FIKS_CO2[k][t] - FIKM_CO2[k][t] + JK_CO2[k][t]
            QI_HPO4[k] = FIKS_HPO4[k][t] - FIKM_HPO4[k][t] + JK_HPO4[k][t]
            QI_H2PO4[k] = FIKS_H2PO4[k][t] - FIKM_H2PO4[k][t] + JK_H2PO4[k][t]
            QI_UREA[k] = FIKS_UREA[k][t] - FIKM_UREA[k][t] + JK_UREA[k][t]
            QI_NH3[k] = FIKS_NH3[k][t] - FIKM_NH3[k][t] + JK_NH3[k][t]
            QI_NH4[k] = FIKS_NH4[k][t] - FIKM_NH4[k][t] + JK_NH4[k][t]
            QI_H[k] = FIKS_H[k][t] - FIKM_H[k][t] + JK_H[k][t]
            QI_HCO2[k] = FIKS_HCO2[k][t] - FIKM_HCO2[k][t] + JK_HCO2[k][t]
            QI_H2CO2[k] = FIKS_H2CO2[k][t] - FIKM_H2CO2[k][t] + JK_H2CO2[k][t]
            QI_GLUC[k] = FIKS_GLUC[k][t] - FIKM_GLUC[k][t] + JK_GLUC[k][t]



    PHIE_VLM = PHIScale(PHIE_VLM , Scale)
    QE_NA = PHIScale(QE_NA, Scale)
    QE_K = PHIScale(QE_K, Scale)
    QE_CL = PHIScale(QE_CL, Scale)
    QE_HCO3 = PHIScale(QE_HCO3, Scale)
    QE_H2CO3 = PHIScale(QE_H2CO3, Scale)
    QE_CO2 = PHIScale(QE_CO2, Scale)
    QE_HPO4 = PHIScale(QE_HPO4, Scale)
    QE_H2PO4 = PHIScale(QE_H2PO4, Scale)
    QE_UREA = PHIScale(QE_UREA, Scale)
    QE_NH3 = PHIScale(QE_NH3, Scale)
    QE_NH4 = PHIScale(QE_NH4, Scale)
    QE_H = PHIScale(QE_H, Scale)
    QE_HCO2 = PHIScale(QE_HCO2, Scale)
    QE_H2CO2 = PHIScale(QE_H2CO2, Scale)
    QE_GLUC = PHIScale(QE_GLUC, Scale)

    PHII_VLM[k] = PHIScale(PHII_VLM[k] , Scale)
    QI_NA[k] = PHIScale(QI_NA[k], Scale)
    QI_K[k] = PHIScale(QI_K[k], Scale)
    QI_CL[k] = PHIScale(QI_CL[k], Scale)
    QI_HCO3[k] = PHIScale(QI_HCO3[k], Scale)
    QI_H2CO3[k] = PHIScale(QI_H2CO3[k], Scale)
    QI_CO2[k] = PHIScale(QI_CO2[k], Scale)
    QI_HPO4[k] = PHIScale(QI_HPO4[k], Scale)
    QI_H2PO4[k] = PHIScale(QI_H2PO4[k], Scale)
    QI_UREA[k] = PHIScale(QI_UREA[k], Scale)
    QI_NH3[k] = PHIScale(QI_NH3[k], Scale)
    QI_NH4[k] = PHIScale(QI_NH4[k], Scale)
    QI_H[k] = PHIScale(QI_H[k], Scale)
    QI_HCO2[k] = PHIScale(QI_HCO2[k], Scale)
    QI_H2CO2[k] = PHIScale(QI_H2CO2[k], Scale)
    QI_GLUC[k] = PHIScale(QI_GLUC[k], Scale)

# NOW SET THE PHIS IN TERMS OF THE SOLUTE GENERATION RATES
# FIRST THE NON-REACTIVE SPECIES

    PHI[1] = PHIE_EN
    PHI[2] = PHIE_VLM

    PHIE_NA = QE_NA
    PHI[3] = PHIE_NA

    PHIE_K = QE_K
    PHI[4] = PHIE_K

    PHIE_CL = QE_CL
    PHI[5] = PHIE_CL

    PHIE_UREA = QE_UREA
    PHI[11] = PHIE_UREA

    PHIE_GLUC = QE_GLUC
    PHI[16] = PHIE_GLUC
    for k in range(0):
        SOLS=15
        kk=SOLS+1
        PHII_NA = QI_NA[k]
        PHI[3+kk] = PHII_NA

        PHII_K = QI_K[k]
        PHI[4+kk] = PHII_K

        PHII_CL = QI_CL[k]
        PHI[5+kk] = PHII_CL

        PHII_UREA = QI_UREA[k]
        PHI[11+kk] = PHII_UREA

        PHII_GLUC = QI_GLUC[k]
        PHI[16+kk] = PHII_GLUC

# CO2, FORMATE, PHOSPHATE, AND AMMONIA CONTENT:
    PHI[8] = QE_HCO3 + QE_H2CO3 + QE_CO2
    PHI[9] = QE_HPO4 + QE_H2PO4
    PHI[12] = QE_NH3 + QE_NH4
    PHI[14] = QE_HCO2 + QE_H2CO2

    for k in range(0):
        PHI[8+kk] = QI_HCO3[k] + QI_H2CO3[k] + QI_CO2[k]
        PHI[9+kk] = QI_HPO4[k] + QI_H2PO4[k]
        PHI[12+kk] = QI_NH3[k] + QI_NH4[k]- 1e6*QIAMM
        PHI[14+kk] = QI_HCO2[k] + QI_H2CO2[k]



#  FORMATE, PHOSPHATE, AND AMMMONIA PH EQUILIBRIUM:
    PHIE_H2PO4 = EBUF(CE_H[t], PKP, CE_HPO4[t], CE_H2PO4[t])
    PHI[10] = PHIE_H2PO4
    PHIE_NH4 = EBUF(CE_H[t], PKN, CE_NH3[t], CE_NH4[t])
    PHI[13] = PHIE_NH4
    PHIE_H2CO2 = EBUF(CE_H[t], PKF, CE_HCO2[t], CE_H2CO2[t])
    PHI[15] = PHIE_H2CO2
#PHIE_H2CO3 = EBUF(CE_H[t], PKC, CE_HCO3[t], CE_H2CO3[t])
    for k in range(0):
        kk=SOLS+1
        PHII_H2PO4 = EBUF(CI_H[k][t], PKP, CI_HPO4[k][t], CI_H2PO4[k][t])
        PHI[10+kk] = PHII_H2PO4
        PHII_NH4 = EBUF(CI_H[k][t], PKN, CI_NH3[k][t], CI_NH4[k][t])
        PHI[13+kk] = PHII_NH4
        PHII_H2CO2 = EBUF(CI_H[k][t], PKF, CI_HCO2[k][t], CI_H2CO2[k][t])
        PHI[15+kk] = PHII_H2CO2

#  HYDRATION AND DHYDRATION OF CO2
    PHIE_CO2 = QE_CO2 + 1.e6*CHVL[t]*(KHY_4*CE_CO2[t] - KDHY_4*CE_H2CO3[t])
    PHI[7] = PHIE_CO2
    for k in range(0):
        kk = SOLS + 1
        PHII_CO2 = QI_CO2[k] + 1.e6 * CLVL[k][t] * (KHY * CI_CO2[k][t] - KDHY * CI_H2CO3[k][t])
        PHI[7+kk] = PHII_CO2

# THE ERROR TERM FOR BICARBONATE GENERATION IS REPLACED BY CONSERVATION
# OF CHARGE IN THE BUFFER REACTIONS.

    PHIE_H2CO3 = - QE_HCO3 + QE_H2PO4 + QE_NH4 + QE_H + QE_H2CO2
    PHI[6] = PHIE_H2CO3

    for k in range(0):
        kk = SOLS + 1
        PHII_H2CO3 = - QI_HCO3[k] + QI_H2PO4[k] + QI_NH4[k] + QI_H[k] + QI_H2CO2[k]
        PHI[6+kk] = PHII_H2CO3

# CELL BUFFER CONTENT AND PH EQUILIBRIUM

    for k in range(0):
        PHI[3+2*SOLS]=CBUF[k][t]+HCBUF[k][t]-(TBUF[k]*CLVL0[k]/CLVL[k][t])
        PHI[4+2*SOLS]=PKB[k]-PKP- math.log10((CI_HPO4[k][t]*HCBUF[k][t])/(CI_H2PO4[k][t]*CBUF[k][t]))








