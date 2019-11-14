import math
import numpy as np
from scipy.optimize import *
import sys

from PCT.PCT_GLOB import CHVL0, F, RTE, NP, KNH4, LHP
from timeit import Timer

sys.path.append('../epithelia/')
from PCT.PCT_GLOB import *

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
    # ''' Logarithm mean membrane solute concentration '''
    import math
    math.log(sys.float_info.min)
    math.log(sys.float_info.min * sys.float_info.epsilon)
    if C_a/C_b > 0 and C_a-C_b != 0:
        return (C_a-C_b)/(math.log10(C_a/C_b))
    else:
        return C_b


def EPS(C, Z, V):
    # '''Electrochemical Potential of Species i'''
    import math
    math.log(sys.float_info.min)
    math.log(sys.float_info.min * sys.float_info.epsilon)
    if C > 0 and C!=0:
        return RTE * math.log10(C) + Z * F * V * 1.e-6
    else:
        return Z*F*V*1.e-6


def Zab(Z, Va, Vb):
        return Z * F * (Va - Vb) * 1.e-6 / RTE

def phiScale(phi, Scale):
    return phi * Scale


T = 1
# t = np.linspace(0, 2, T)
# DT = (2 - 0) / (T - 1)
RTAU = 1#DT
VE = -0.0100
PE = -23.1
CE_NA = 0.1403
CE_K = 0.00466
CE_CL = 0.11200
CE_HCO3 = 0.0256
CE_H2CO3 = 0.00000436
CE_CO2 = 0.00149
CE_HPO4 = 0.00298
CE_H2PO4 = 0.00086
CE_UREA = 0.00491
CE_NH3 = 0.0000027
CE_NH4 = 0.00018
CE_HCO2 = 0.00077
CE_H2CO2 = 0.000000204
CE_GLUC = 0.00779
VI = -55.6
PI = 15.0
CI_NA = 0.01900
CI_K = 0.1381
CI_CL = 0.0163
CI_HCO3 = 0.025000000
CI_H2CO3 = 0.00000436
CI_CO2 = 0.00149
CI_HPO4 = 0.008500
CI_H2PO4 = 0.0025200
CI_UREA = 0.00496
CI_NH3 = 0.00000348
CI_NH4 = 0.0002300
CI_HCO2 = 0.000520029
CI_H2CO2 = 0.000000091
CI_GLUC = 0.015000000
CE_H = 4.59e-11
CI_H = 4.69e-11
CM_H = 4.95e-11
CS_H = 4.95e-11
PS = 0
VS = 0
AE = 0.0
CHVL = 0.7000e-04
CLVL = 0.1000e-02
L = 0.1000e-01
IMP = 0.6000e-01
HCBUF = 0.0
CBUF = 0.0

RM = 0.1060e-02
AM = 0

LCHM = 0.3
IMPS = 0.9
IMPM = 0.9

FEVS = 0.01
FIVM = 0.1
FIVS = 0
JV = 0

FEVM = 0.9
FEKM_NA = 0.01
FEKM_K = 0.01
FEKM_CL = 0.01
FEKM_HCO3 = 0.01
FEKM_H2CO3 = 0.01
FEKM_CO2 = 0.01
FEKM_HPO4 = 0.01
FEKM_H2PO4 = 0.01
FEKM_UREA = 0.01
FEKM_NH3 = 0.01
FEKM_NH4 = 0.01
FEKM_H = 0.01
FEKM_HCO2 = 0.01
FEKM_H2CO2 = 0.01
FEKM_GLUC = 0.01

FEKS_NA = 0.01
FEKS_K = 0.01
FEKS_CL = 0.01
FEKS_HCO3 = 0.01
FEKS_H2CO3 = 0.01
FEKS_CO2 = 0.01
FEKS_HPO4 = 0.01
FEKS_H2PO4 = 0.01
FEKS_UREA = 0.01
FEKS_NH3 = 0.01
FEKS_NH4 = 0.01
FEKS_H = 0.01
FEKS_HCO2 = 0.01
FEKS_H2CO2 = 0.01
FEKS_GLUC = 0.01

FIKM_NA = 0.01
FIKM_K = 0.01
FIKM_CL = 0.01
FIKM_HCO3 = 0.01
FIKM_H2CO3 = 0.01
FIKM_CO2 = 0.01
FIKM_HPO4 = 0.01
FIKM_H2PO4 = 0.01
FIKM_UREA = 0.01
FIKM_NH3 = 0.01
FIKM_NH4 = 0.01
FIKM_H = 0.01
FIKM_HCO2 = 0.01
FIKM_H2CO2 = 0.01
FIKM_GLUC = 0.01

FIKS_NA = 0.01
FIKS_K = 0.01
FIKS_CL = 0.01
FIKS_HCO3 = 0.01
FIKS_H2CO3 = 0.01
FIKS_CO2 = 0.01
FIKS_HPO4 = 0.01
FIKS_H2PO4 = 0.01
FIKS_UREA = 0.01
FIKS_NH3 = 0.01
FIKS_NH4 = 0.01
FIKS_H = 0.01
FIKS_HCO2 = 0.01
FIKS_H2CO2 = 0.01
FIKS_GLUC = 0.01

JK_NA = 0.01
JK_K = 0.01
JK_CL = 0.01
JK_HCO3 = 0.01
JK_H2CO3 = 0.01
JK_CO2 = 0.01
JK_HPO4 = 0.01
JK_H2PO4 = 0.01
JK_UREA = 0.01
JK_NH3 = 0.01
JK_NH4 = 0.01
JK_H = 0.01
JK_HCO2 = 0.01
JK_H2CO2 = 0.01
JK_GLUC = 0.01
phi_CUR = [None] * 35
phi= [None] * 35

def EQS(i,GUESS):
    # COMPUTE THE RELEVANT VOLUMES

    VE = GUESS[0]
    PE = GUESS[1]
    CE_NA = GUESS[2]
    CE_K = GUESS[3]
    CE_CL = GUESS[4]
    CE_HCO3 = GUESS[5]
    CE_H2CO3 = GUESS[6]
    CE_CO2 = GUESS[7]
    CE_HPO4 = GUESS[8]
    CE_H2PO4 = GUESS[9]
    CE_UREA = GUESS[10]
    CE_NH3 = GUESS[11]
    CE_NH4 = GUESS[12]
    CE_HCO2 = GUESS[13]
    CE_H2CO2 = GUESS[14]
    CE_GLUC = GUESS[15]
    VI = GUESS[16]
    IMP = GUESS[17]
    CI_NA = GUESS[18]
    CI_K = GUESS[19]
    CI_CL = GUESS[20]
    CI_HCO3 = GUESS[21]
    CI_H2CO3 = GUESS[22]
    CI_CO2 = GUESS[23]
    CI_HPO4 = GUESS[24]
    CI_H2PO4 = GUESS[25]
    CI_UREA = GUESS[26]
    CI_NH3 = GUESS[27]
    CI_NH4 = GUESS[28]
    CI_HCO2 = GUESS[29]
    CI_H2CO2 = GUESS[30]
    CI_GLUC = GUESS[31]
    CBUF = GUESS[32]
    HCBUF = GUESS[33]
    VM = GUESS[34]

    AE = AE0 * (MUA * (PE - PM))
    AE = AE0 if AE < AE0 else AE
    CHVL = CHVL0 * (1.0 + MUV * (PE - PM))
    CHVL = CHVL0 if CHVL < CHVL0 else CHVL

    L = CHVL
    LCHE = PKC + np.log10(CE_HCO3)-np.log10( CE_H2CO3 )
    CE_H= 10. ** (- LCHE)
    CLVL= CLVL0 * ( IMP0 / IMP)
    CLVL= CLVL0 if IMP == 0 else CLVL

    L= L + CLVL
    PI= PM

    LCHI = PKC + np.log10(CI_HCO3) - np.log10(CI_H2CO3)
    CI_H= 10. ** (-LCHI)
    FEVM = AME * LPME * (PM - PE - RT * IMPM) / RT
    FEVS= AE * LPES * (RT * IMPS + PE - PS) / RT
    FIVM= AMI * LPMI * (RT * IMP - RT * IMPM + PM - PI) / RT
    FIVS= AIS * LPIS * (PI - PS + RT * IMPS - RT * IMP) / RT
    JV= LPIS * AIE * (PI - PE - RT * IMP) / RT

    FEVM= FEVM + AME * LPME * (
            SME_NA * (CE_NA - CM_NA) + SME_K * (
            CE_K - CM_K) + SME_CL * (
                    CE_CL - CM_CL) + SME_HCO3 * (
                    CE_HCO3 - CM_HCO3)
            + SME_H2CO3 * (CE_H2CO3 - CM_H2CO3) + SME_CO2 * (
                    CE_CO2 - CM_CO2) + SME_HPO4 * (
                    CE_HPO4 - CM_HPO4) + SME_H2PO4 * (
                    CE_H2PO4 - CM_H2PO4)
            + SME_UREA * (CE_UREA - CM_UREA) + SME_NH3 * (
                    CE_NH3 - CM_NH3) + SME_NH4 * (
                    CE_NH4 - CM_NH4) + SME_H * (
                    CE_H - CM_H) + SME_HCO2 * (
                    CE_HCO2 - CM_HCO2)
            + SME_H2CO2 * (CE_H2CO2 - CM_H2CO2) + SME_GLUC * (
                    CE_GLUC - CM_GLUC))


    FEVS= FEVS + AE * LPES * (
            SES_NA * (CS_NA - CE_NA) + SES_K * (
            CS_K - CE_K) + SES_CL * (
                    CS_CL - CE_CL) + SES_HCO3 * (
                    CS_HCO3 - CE_HCO3)
            + SES_H2CO3 * (CS_H2CO3 - CE_H2CO3) + SES_CO2 * (
                    CS_CO2 - CE_CO2) + SES_HPO4 * (
                    CS_HPO4 - CE_HPO4) + SES_H2PO4 * (
                    CS_H2PO4 - CE_H2PO4)
            + SES_UREA * (CS_UREA - CE_UREA) + SES_NH3 * (
                    CS_NH3 - CE_NH3) + SES_NH4 * (
                    CS_NH4 - CE_NH4) + SES_H * (
                    CS_H - CE_H) + SES_HCO2 * (
                    CS_HCO2 - CE_HCO2)
            + SES_H2CO2 * (CS_H2CO2 - CE_H2CO2) + SES_GLUC * (
                    CS_GLUC - CE_GLUC))

    FIVM = FIVM + AMI * LPMI * (
            SMI_NA * (CI_NA - CM_NA) + SMI_K * (
            CI_K - CM_K) + SMI_CL * (
                    CI_CL - CM_CL) + SMI_HCO3 * (
                    CI_HCO3 - CM_HCO3)
            + SMI_H2CO3 * (CI_H2CO3 - CM_H2CO3) + SMI_CO2 * (
                    CI_CO2 - CM_CO2) + SMI_HPO4 * (
                    CI_HPO4 - CM_HPO4) + SMI_H2PO4 * (
                    CI_H2PO4 - CM_H2PO4)
            + SMI_UREA * (CI_UREA - CM_UREA) + SMI_NH3 * (
                    CI_NH3 - CM_NH3) + SMI_NH4 * (
                    CI_NH4 - CM_NH4) + SMI_HCO2 * (
                    CI_HCO2 - CM_HCO2)
            + SMI_H2CO2 * (CI_H2CO2 - CM_H2CO2) + SMI_GLUC * (
                    CI_GLUC - CM_GLUC))

    FIVS = FIVS + AIS * LPIS * (
            SIS_NA * (CS_NA - CI_NA) + SIS_K * (
            CS_K - CI_K) + SIS_CL * (
                    CS_CL - CI_CL) + SIS_HCO3 * (
                    CS_HCO3 - CI_HCO3)
            + SIS_H2CO3 * (CS_H2CO3 - CI_H2CO3) + SIS_CO2 * (
                    CS_CO2 - CI_CO2) + SIS_HPO4 * (
                    CS_HPO4 - CI_HPO4) + SIS_H2PO4 * (
                    CS_H2PO4 - CI_H2PO4)
            + SIS_UREA * (CS_UREA - CI_UREA) + SIS_NH3 * (
                    CS_NH3 - CI_NH3) + SIS_NH4 * (
                    CS_NH4 - CI_NH4) + SIS_H * (
                    CS_H - CI_H) + SIS_HCO2 * (
                    CS_HCO2 - CI_HCO2)
            + SIS_H2CO2 * (CS_H2CO2 - CI_H2CO2) + SIS_GLUC * (
                    CS_GLUC - CI_GLUC))

    JV = JV + LPIS * AIE * (
            SIS_NA * (CE_NA - CI_NA) + SIS_K * (
            CE_K - CI_K) + SIS_CL * (
                    CE_CL - CI_CL) + SIS_HCO3 * (
                    CE_HCO3 - CI_HCO3)
            + SIS_H2CO3 * (CE_H2CO3 - CI_H2CO3) + SIS_CO2 * (
                    CE_CO2 - CI_CO2) + SIS_HPO4 * (
                    CE_HPO4 - CI_HPO4) + SIS_H2PO4 * (
                    CE_H2PO4 - CI_H2PO4)
            + SIS_UREA * (CE_UREA - CI_UREA) + SIS_NH3 * (
                    CE_NH3 - CI_NH3) + SIS_NH4 * (
                    CE_NH4 - CI_NH4) + SIS_HCO2 * (
                    CE_HCO2 - CI_HCO2)
            + SIS_H2CO2 * (CE_H2CO2 - CI_H2CO2) + SIS_GLUC * (
                    CE_GLUC - CI_GLUC))
    CME_NA = LMMS(CE_NA, CM_NA)
    CME_K = LMMS(CE_K, CM_K)
    CME_CL = LMMS(CE_CL, CM_CL)
    CME_HCO3 = LMMS(CE_HCO3, CM_HCO3)
    CME_H2CO3 = LMMS(CE_H2CO3, CM_H2CO3)
    CME_CO2 = LMMS(CE_CO2, CM_CO2)
    CME_HPO4 = LMMS(CE_HPO4, CM_HPO4)
    CME_H2PO4 = LMMS(CE_H2PO4, CM_H2PO4)
    CME_UREA = LMMS(CE_UREA, CM_UREA)
    CME_NH3 = LMMS(CE_NH3, CM_NH3)
    CME_NH4 = LMMS(CE_NH4, CM_NH4)
    CME_H = LMMS(CE_H, CM_H)
    CME_HCO2 = LMMS(CE_HCO2, CM_HCO2)
    CME_H2CO2 = LMMS(CE_H2CO2, CM_H2CO2)
    CME_GLUC = LMMS(CE_GLUC, CM_GLUC)

    CES_NA = LMMSC(CE_NA, CS_NA)
    CES_K = LMMSC(CE_K, CS_K)
    CES_CL = LMMSC(CE_CL, CS_CL)
    CES_HCO3 = LMMSC(CE_HCO3, CS_HCO3)
    CES_H2CO3 = LMMSC(CE_H2CO3, CS_H2CO3)
    CES_CO2 = LMMSC(CE_CO2, CS_CO2)
    CES_HPO4 = LMMSC(CE_HPO4, CS_HPO4)
    CES_H2PO4 = LMMSC(CE_H2PO4, CS_H2PO4)
    CES_UREA = LMMSC(CE_UREA, CS_UREA)
    CES_NH3 = LMMSC(CE_NH3, CS_NH3)
    CES_NH4 = LMMSC(CE_NH4, CS_NH4)
    CES_H = LMMSC(CE_H, CS_H)
    CES_HCO2 = LMMSC(CE_HCO2, CS_HCO2)
    CES_H2CO2 = LMMSC(CE_H2CO2, CS_H2CO2)
    CES_GLUC = LMMSC(CE_GLUC, CS_GLUC)

    CMI_NA= LMMSC(CI_NA, CM_NA)
    CMI_K= LMMSC(CI_K, CM_K)
    CMI_CL= LMMSC(CI_CL, CM_CL)
    CMI_HCO3= LMMSC(CI_HCO3, CM_HCO3)
    CMI_H2CO3= LMMSC(CI_H2CO3, CM_H2CO3)
    CMI_CO2= LMMSC(CI_CO2, CM_CO2)
    CMI_HPO4= LMMSC(CI_HPO4, CM_HPO4)
    CMI_H2PO4= LMMSC(CI_H2PO4, CM_H2PO4)
    CMI_UREA= LMMSC(CI_UREA, CM_UREA)
    CMI_NH3= LMMSC(CI_NH3, CM_NH3)
    CMI_NH4= LMMSC(CI_NH4, CM_NH4)
    CMI_H= LMMSC(CI_H, CM_H)
    CMI_HCO2= LMMSC(CI_HCO2, CM_HCO2)
    CMI_H2CO2= LMMSC(CI_H2CO2, CM_H2CO2)
    CMI_GLUC= LMMSC(CI_GLUC, CM_GLUC)

    CIS_NA= LMMSC(CI_NA, CS_NA)
    CIS_K= LMMSC(CI_K, CS_K)
    CIS_CL= LMMSC(CI_CL, CS_CL)
    CIS_HCO3= LMMSC(CI_HCO3, CS_HCO3)
    CIS_H2CO3= LMMSC(CI_H2CO3, CS_H2CO3)
    CIS_CO2= LMMSC(CI_CO2, CS_CO2)
    CIS_HPO4= LMMSC(CI_HPO4, CS_HPO4)
    CIS_H2PO4= LMMSC(CI_H2PO4, CS_H2PO4)
    CIS_UREA= LMMSC(CI_UREA, CS_UREA)
    CIS_NH3= LMMSC(CI_NH3, CS_NH3)
    CIS_NH4= LMMSC(CI_NH4, CS_NH4)
    CIS_H= LMMSC(CI_H, CS_H)
    CIS_HCO2= LMMSC(CI_HCO2, CS_HCO2)
    CIS_H2CO2= LMMSC(CI_H2CO2, CS_H2CO2)
    CIS_GLUC= LMMSC(CI_GLUC, CS_GLUC)

    CIE_NA= LMMSC(CI_NA, CE_NA)
    CIE_K= LMMSC(CI_K, CE_K)
    CIE_CL= LMMSC(CI_CL, CE_CL)
    CIE_HCO3= LMMSC(CI_HCO3, CE_HCO3)
    CIE_H2CO3= LMMSC(CI_H2CO3, CE_H2CO3)
    CIE_CO2= LMMSC(CI_CO2, CE_CO2)
    CIE_HPO4= LMMSC(CI_HPO4, CE_HPO4)
    CIE_H2PO4= LMMSC(CI_H2PO4, CE_H2PO4)
    CIE_UREA= LMMSC(CI_UREA, CE_UREA)
    CIE_NH3= LMMSC(CI_NH3, CE_NH3)
    CIE_NH4= LMMSC(CI_NH4, CE_NH4)
    CIE_H= LMMSC(CI_H, CE_H)
    CIE_HCO2= LMMSC(CI_HCO2, CE_HCO2)
    CIE_H2CO2= LMMSC(CI_H2CO2, CE_H2CO2)
    CIE_GLUC= LMMSC(CI_GLUC, CE_GLUC)

    # CONVECTIVE FLUXES
    FEKM_NA = FEVM * (1.00 - SME_NA) * CME_NA
    FEKM_K = FEVM * (1.00 - SME_K) * CME_K
    FEKM_CL = FEVM * (1.00 - SME_CL) * CME_CL
    FEKM_HCO3 = FEVM * (1.00 - SME_HCO3) * CME_HCO3
    FEKM_H2CO3 = FEVM * (1.00 - SME_H2CO3) * CME_H2CO3
    FEKM_CO2 = FEVM * (1.00 - SME_CO2) * CME_CO2
    FEKM_HPO4 = FEVM * (1.00 - SME_HPO4) * CME_HPO4
    FEKM_H2PO4 = FEVM * (1.00 - SME_H2PO4) * CME_H2PO4
    FEKM_UREA = FEVM * (1.00 - SME_UREA) * CME_UREA
    FEKM_NH3 = FEVM * (1.00 - SME_NH3) * CME_NH3
    FEKM_NH4 = FEVM * (1.00 - SME_NH4) * CME_NH4
    FEKM_H = FEVM * (1.00 - SME_H) * CME_H
    FEKM_HCO2 = FEVM * (1.00 - SME_HCO2) * CME_HCO2
    FEKM_H2CO2 = FEVM * (1.00 - SME_H2CO2) * CME_H2CO2
    FEKM_GLUC = FEVM * (1.00 - SME_GLUC) * CME_GLUC

    FEKS_NA = FEVS * (1.00 - SES_NA) * CES_NA
    FEKS_K = FEVS * (1.00 - SES_K) * CES_K
    FEKS_CL = FEVS * (1.00 - SES_CL) * CES_CL
    FEKS_HCO3 = FEVS * (1.00 - SES_HCO3) * CES_HCO3
    FEKS_H2CO3 = FEVS * (1.00 - SES_H2CO3) * CES_H2CO3
    FEKS_CO2 = FEVS * (1.00 - SES_CO2) * CES_CO2
    FEKS_HPO4 = FEVS * (1.00 - SES_HPO4) * CES_HPO4
    FEKS_H2PO4 = FEVS * (1.00 - SES_H2PO4) * CES_H2PO4
    FEKS_UREA = FEVS * (1.00 - SES_UREA) * CES_UREA
    FEKS_NH3 = FEVS * (1.00 - SES_NH3) * CES_NH3
    FEKS_NH4 = FEVS * (1.00 - SES_NH4) * CES_NH4
    FEKS_H = FEVS * (1.00 - SES_H) * CES_H
    FEKS_HCO2 = FEVS * (1.00 - SES_HCO2) * CES_HCO2
    FEKS_H2CO2 = FEVS * (1.00 - SES_H2CO2) * CES_H2CO2
    FEKS_GLUC = FEVS * (1.00 - SES_GLUC) * CES_GLUC

    FIKM_NA = FIVM * (1.00 - SMI_NA)*CMI_NA
    FIKM_K = FIVM * (1.00 - SMI_K)*CMI_K
    FIKM_CL = FIVM * (1.00 - SMI_CL)*CMI_CL
    FIKM_HCO3 = FIVM * (1.00 - SMI_HCO3)*CMI_HCO3
    FIKM_H2CO3 = FIVM * (1.00 - SMI_H2CO3)*CMI_H2CO3
    FIKM_CO2 = FIVM * (1.00 - SMI_CO2)*CMI_CO2
    FIKM_HPO4 = FIVM * (1.00 - SMI_HPO4)*CMI_HPO4
    FIKM_H2PO4 = FIVM * (1.00 - SMI_H2PO4)*CMI_H2PO4
    FIKM_UREA = FIVM * (1.00 - SMI_UREA)*CMI_UREA
    FIKM_NH3 = FIVM * (1.00 - SMI_NH3)*CMI_NH3
    FIKM_NH4 = FIVM * (1.00 - SMI_NH4)*CMI_NH4
    FIKM_H = FIVM * (1.00 - SMI_H)*CMI_H
    FIKM_HCO2 = FIVM * (1.00 - SMI_HCO2)*CMI_HCO2
    FIKM_H2CO2 = FIVM * (1.00 - SMI_H2CO2)*CMI_H2CO2
    FIKM_GLUC = FIVM * (1.00 - SMI_GLUC)*CMI_GLUC

    FIKS_NA = FIVS * (1.00 - SIS_NA)*CIS_NA
    FIKS_K = FIVS * (1.00 - SIS_K)*CIS_K
    FIKS_CL = FIVS * (1.00 - SIS_CL)*CIS_CL
    FIKS_HCO3 = FIVS * (1.00 - SIS_HCO3)*CIS_HCO3
    FIKS_H2CO3 = FIVS * (1.00 - SIS_H2CO3)*CIS_H2CO3
    FIKS_CO2 = FIVS * (1.00 - SIS_CO2)*CIS_CO2
    FIKS_HPO4 = FIVS * (1.00 - SIS_HPO4)*CIS_HPO4
    FIKS_H2PO4 = FIVS * (1.00 - SIS_H2PO4)*CIS_H2PO4
    FIKS_UREA = FIVS * (1.00 - SIS_UREA)*CIS_UREA
    FIKS_NH3 = FIVS * (1.00 - SIS_NH3)*CIS_NH3
    FIKS_NH4 = FIVS * (1.00 - SIS_NH4)*CIS_NH4
    FIKS_H = FIVS * (1.00 - SIS_H)*CIS_H
    FIKS_HCO2 = FIVS * (1.00 - SIS_HCO2)*CIS_HCO2
    FIKS_H2CO2 = FIVS * (1.00 - SIS_H2CO2)*CIS_H2CO2
    FIKS_GLUC = FIVS * (1.00 - SIS_GLUC)*CIS_GLUC

    JK_NA = JV * (1.00 - SIS_NA)*CIE_NA
    JK_K = JV * (1.00 - SIS_K)*CIE_K
    JK_CL = JV * (1.00 - SIS_CL)*CIE_CL
    JK_HCO3 = JV * (1.00 - SIS_HCO3)*CIE_HCO3
    JK_H2CO3 = JV * (1.00 - SIS_H2CO3)*CIE_H2CO3
    JK_CO2 = JV * (1.00 - SIS_CO2)*CIE_CO2
    JK_HPO4 = JV * (1.00 - SIS_HPO4)*CIE_HPO4
    JK_H2PO4 = JV * (1.00 - SIS_H2PO4)*CIE_H2PO4
    JK_UREA = JV * (1.00 - SIS_UREA)*CIE_UREA
    JK_NH3 = JV * (1.00 - SIS_NH3)*CIE_NH3
    JK_NH4 = JV * (1.00 - SIS_NH4)*CIE_NH4
    JK_H = JV * (1.00 - SIS_H)*CIE_H
    JK_HCO2 = JV * (1.00 - SIS_HCO2)*CIE_HCO2
    JK_H2CO2 = JV * (1.00 - SIS_H2CO2)*CIE_H2CO2
    JK_GLUC = JV * (1.00 - SIS_GLUC)*CIE_GLUC

    # GOLDMAN  FLUXES
    FEKM_NA = FEKM_NA + GOLDMAN(HME_NA, AME, Z_NA, VM, VE, CM_NA,
                                            CE_NA, 1)
    FEKM_K = FEKM_K + GOLDMAN(HME_K, AME, Z_K, VM, VE, CM_K, CE_K, 1)
    FEKM_CL = FEKM_CL + GOLDMAN(HME_CL, AME, Z_CL, VM, VE, CM_CL, CE_CL, 1)
    FEKM_HCO3 = FEKM_HCO3 + GOLDMAN(HME_HCO3, AME, Z_HCO3, VM, VE, CM_HCO3, CE_HCO3,
                                                1)
    FEKM_H2CO3 = FEKM_H2CO3 + GOLDMAN(HME_H2CO3, AME, Z_H2CO3, VM, VE, CM_H2CO3,
                                                  CE_H2CO3, 1)
    FEKM_CO2 = FEKM_CO2 + GOLDMAN(HME_CO2, AME, Z_CO2, VM, VE, CM_CO2, CE_CO2, 1)
    FEKM_HPO4 = FEKM_HPO4 + GOLDMAN(HME_HPO4, AME, Z_HPO4, VM, VE, CM_HPO4,
                                                CE_HPO4, 1)
    FEKM_H2PO4 = FEKM_H2PO4 + GOLDMAN(HME_H2PO4, AME, Z_H2PO4, VM, VE, CM_H2PO4,
                                                  CE_H2PO4, 1)
    FEKM_UREA = FEKM_UREA + GOLDMAN(HME_UREA, AME, Z_UREA, VM, VE, CM_UREA, CE_UREA,
                                                1)
    FEKM_NH3 = FEKM_NH3 + GOLDMAN(HME_NH3, AME, Z_NH3, VM, VE, CM_NH3, CE_NH3, 1)
    FEKM_NH4 = FEKM_NH4 + GOLDMAN(HME_NH4, AME, Z_NH4, VM, VE, CM_NH4, CE_NH4, 1)
    FEKM_H = FEKM_H + GOLDMAN(HME_H, AME, Z_H, VM, VE, CM_H, CE_H, 1)
    FEKM_HCO2 = FEKM_HCO2 + GOLDMAN(HME_HCO2, AME, Z_HCO2, VM, VE, CM_HCO2, CE_HCO2,
                                                1)
    FEKM_H2CO2 = FEKM_H2CO2 + GOLDMAN(HME_H2CO2, AME, Z_H2CO2, VM, VE, CM_H2CO2,
                                                  CE_H2CO2, 1)
    FEKM_GLUC = FEKM_GLUC + GOLDMAN(HME_GLUC, AME, Z_GLUC, VM, VE, CM_GLUC, CE_GLUC,1)

    FEKS_NA = FEKS_NA + GOLDMAN(HES_NA, AE, Z_NA, VE, VS, CE_NA,
                                            CS_NA, 1)
    FEKS_K = FEKS_K + GOLDMAN(HES_K, AE, Z_K, VE, VS, CE_K, CS_K, 1)
    FEKS_CL = FEKS_CL + GOLDMAN(HES_CL, AE, Z_CL, VE, VS, CE_CL, CS_CL, 1)
    FEKS_HCO3 = FEKS_HCO3 + GOLDMAN(HES_HCO3, AE, Z_HCO3, VE, VS, CE_HCO3,
                                                CS_HCO3, 1)
    FEKS_H2CO3 = FEKS_H2CO3 + GOLDMAN(HES_H2CO3, AE, Z_H2CO3, VE, VS, CE_H2CO3,
                                                  CS_H2CO3, 1)
    FEKS_CO2 = FEKS_CO2 + GOLDMAN(HES_CO2, AE, Z_CO2, VE, VS, CE_CO2, CS_CO2, 1)
    FEKS_HPO4 = FEKS_HPO4 + GOLDMAN(HES_HPO4, AE, Z_HPO4, VE, VS, CE_HPO4,
                                                CS_HPO4, 1)
    FEKS_H2PO4 = FEKS_H2PO4 + GOLDMAN(HES_H2PO4, AE, Z_H2PO4, VE, VS, CE_H2PO4,
                                                  CS_H2PO4, 1)
    FEKS_UREA = FEKS_UREA + GOLDMAN(HES_UREA, AE, Z_UREA, VE, VS, CE_UREA,
                                                CS_UREA, 1)
    FEKS_NH3 = FEKS_NH3 + GOLDMAN(HES_NH3, AE, Z_NH3, VE, VS, CE_NH3, CS_NH3, 1)
    FEKS_NH4 = FEKS_NH4 + GOLDMAN(HES_NH4, AE, Z_NH4, VE, VS, CE_NH4, CS_NH4, 1)
    FEKS_H = FEKS_H + GOLDMAN(HES_H, AE, Z_H, VE, VS, CE_H, CS_H, 1)
    FEKS_HCO2 = FEKS_HCO2 + GOLDMAN(HES_HCO2, AE, Z_HCO2, VE, VS, CE_HCO2,
                                                CS_HCO2, 1)

    FIKM_NA = FIKM_NA + GOLDMAN(HMI_NA, AMI, Z_NA, VM, VI, CM_NA,
                                            CI_NA, 1)
    FIKM_K = FIKM_K + GOLDMAN(HMI_K, AMI, Z_K, VM, VI, CM_K, CI_K, 1)
    FIKM_CL = FIKM_CL + GOLDMAN(HMI_CL, AMI, Z_CL, VM, VI, CM_CL, CI_CL, 1)
    FIKM_HCO3 = FIKM_HCO3 + GOLDMAN(HMI_HCO3, AMI, Z_HCO3, VM, VI, CM_HCO3, CI_HCO3, 1)
    FIKM_H2CO3 = FIKM_H2CO3 + GOLDMAN(HMI_H2CO3, AMI, Z_H2CO3, VM, VI, CM_H2CO3, CI_H2CO3, 1)
    FIKM_CO2 = FIKM_CO2 + GOLDMAN(HMI_CO2, AMI, Z_CO2, VM, VI, CM_CO2, CI_CO2, 1)
    FIKM_HPO4 = FIKM_HPO4 + GOLDMAN(HMI_HPO4, AMI, Z_HPO4, VM, VI, CM_HPO4, CI_HPO4, 1)
    FIKM_H2PO4 = FIKM_H2PO4 + GOLDMAN(HMI_H2PO4, AMI, Z_H2PO4, VM, VI, CM_H2PO4,
                                                  CI_H2PO4, 1)
    FIKM_UREA = FIKM_UREA + GOLDMAN(HMI_UREA, AMI, Z_UREA, VM, VI, CM_UREA, CI_UREA, 1)
    FIKM_NH3 = FIKM_NH3 + GOLDMAN(HMI_NH3, AMI, Z_NH3, VM, VI, CM_NH3, CI_NH3, 1)
    FIKM_NH4 = FIKM_NH4 + GOLDMAN(HMI_NH4, AMI, Z_NH4, VM, VI, CM_NH4, CI_NH4, 1)
    FIKM_H = FIKM_H + GOLDMAN(HMI_H, AMI, Z_H, VM, VI, CM_H, CI_H, 1)
    FIKM_HCO2 = FIKM_HCO2 + GOLDMAN(HMI_HCO2, AMI, Z_HCO2, VM, VI, CM_HCO2, CI_HCO2, 1)
    FIKM_H2CO2 = FIKM_H2CO2 + GOLDMAN(HMI_H2CO2, AMI, Z_H2CO2, VM, VI, CM_H2CO2,
                                                  CI_H2CO2, 1)
    FIKM_GLUC = FIKM_GLUC + GOLDMAN(HMI_GLUC, AMI, Z_GLUC, VM, VI, CM_GLUC, CI_GLUC, 1)
    JK_NA = JK_NA + GOLDMAN(HIS_NA, AIE, Z_NA, VI, VE, CI_NA,
                                        CE_NA, 1)
    JK_K = JK_K + GOLDMAN(HIS_K, AIE, Z_K, VI, VE, CI_K, CE_K, 1)
    JK_CL = JK_CL + GOLDMAN(HIS_CL, AIE, Z_CL, VI, VE, CI_CL, CE_CL, 1)
    JK_HCO3 = JK_HCO3 + GOLDMAN(HIS_HCO3, AIE, Z_HCO3, VI, VE, CI_HCO3, CE_HCO3, 1)
    JK_H2CO3 = JK_H2CO3 + GOLDMAN(HIS_H2CO3, AIE, Z_H2CO3, VI, VE, CI_H2CO3,
                                              CE_H2CO3, 1)
    JK_CO2 = JK_CO2 + GOLDMAN(HIS_CO2, AIE, Z_CO2, VI, VE, CI_CO2, CE_CO2, 1)
    JK_HPO4 = JK_HPO4 + GOLDMAN(HIS_HPO4, AIE, Z_HPO4, VI, VE, CI_HPO4,
                                            CE_HPO4, 1)
    JK_H2PO4 = JK_H2PO4 + GOLDMAN(HIS_H2PO4, AIE, Z_H2PO4, VI, VE, CI_H2PO4,
                                              CE_H2PO4, 1)
    JK_UREA = JK_UREA + GOLDMAN(HIS_UREA, AIE, Z_UREA, VI, VE, CI_UREA, CE_UREA, 1)
    JK_NH3 = JK_NH3 + GOLDMAN(HIS_NH3, AIE, Z_NH3, VI, VE, CI_NH3, CE_NH3, 1)
    JK_NH4 = JK_NH4 + GOLDMAN(HIS_NH4, AIE, Z_NH4, VI, VE, CI_NH4, CE_NH4, 1)
    JK_H = JK_H + GOLDMAN(HIS_H, AIE, Z_H, VI, VE, CI_H, CE_H, 1)
    JK_HCO2 = JK_HCO2 + GOLDMAN(HIS_HCO2, AIE, Z_HCO2, VI, VE, CI_HCO2, CE_HCO2, 1)
    JK_H2CO2 = JK_H2CO2 + GOLDMAN(HIS_H2CO2, AIE, Z_H2CO2, VI, VE, CI_H2CO2,
                                              CE_H2CO2, 1)
    JK_GLUC = JK_GLUC + GOLDMAN(HIS_GLUC, AIE, Z_GLUC, VI, VE, CI_GLUC, CE_GLUC, 1)

    FIKS_NA = FIKS_NA + GOLDMAN(HIS_NA, AIS, Z_NA, VI, VS, CI_NA,
                                            CS_NA, 1)
    FIKS_K = FIKS_K + GOLDMAN(HIS_K, AIS, Z_K, VI, VS, CI_K, CS_K, 1)
    FIKS_CL = FIKS_CL + GOLDMAN(HIS_CL, AIS, Z_CL, VI, VS, CI_CL, CS_CL, 1)
    FIKS_HCO3 = FIKS_HCO3 + GOLDMAN(HIS_HCO3, AIS, Z_HCO3, VI, VS, CI_HCO3, CS_HCO3, 1)
    FIKS_H2CO3 = FIKS_H2CO3 + GOLDMAN(HIS_H2CO3, AIS, Z_H2CO3, VI, VS, CI_H2CO3,
                                                  CS_H2CO3, 1)
    FIKS_CO2 = FIKS_CO2 + GOLDMAN(HIS_CO2, AIS, Z_CO2, VI, VS, CI_CO2, CS_CO2, 1)
    FIKS_HPO4 = FIKS_HPO4 + GOLDMAN(HIS_HPO4, AIS, Z_HPO4, VI, VS, CI_HPO4,
                                                CS_HPO4, 1)
    FIKS_H2PO4 = FIKS_H2PO4 + GOLDMAN(HIS_H2PO4, AIS, Z_H2PO4, VI, VS, CI_H2PO4,
                                                  CS_H2PO4, 1)
    FIKS_UREA = FIKS_UREA + GOLDMAN(HIS_UREA, AIS, Z_UREA, VI, VS, CI_UREA, CS_UREA, 1)
    FIKS_NH3 = FIKS_NH3 + GOLDMAN(HIS_NH3, AIS, Z_NH3, VI, VS, CI_NH3, CS_NH3, 1)
    FIKS_NH4 = FIKS_NH4 + GOLDMAN(HIS_NH4, AIS, Z_NH4, VI, VS, CI_NH4, CS_NH4, 1)
    FIKS_H = FIKS_H + GOLDMAN(HIS_H, AIS, Z_H, VI, VS, CI_H, CS_H, 1)
    FIKS_HCO2 = FIKS_HCO2 + GOLDMAN(HIS_HCO2, AIS, Z_HCO2, VI, VS, CI_HCO2, CS_HCO2, 1)
    FIKS_H2CO2 = FIKS_H2CO2 + GOLDMAN(HIS_H2CO2, AIS, Z_H2CO2, VI, VS, CI_H2CO2,
                                                  CS_H2CO2, 1)
    FIKS_GLUC = FIKS_GLUC + GOLDMAN(HIS_GLUC, AIS, Z_GLUC, VI, VS, CI_GLUC, CS_GLUC, 1)

    # Net Cotransporters
    SGLT = SGLT_MI(CM_NA, CI_NA, CM_GLUC, CI_GLUC, Z_NA, Z_GLUC, VM, VI, AMI, LMI_NAGLUC,
                   1)
    NA_MI_NAGLUC = SGLT[0]
    GLUC_MI_NAGLUC = SGLT[1]
    NAH2PO4 = NAH2PO4_MI(CM_NA, CI_NA, CM_H2PO4, CI_H2PO4, Z_NA, Z_H2PO4, VM, VI, AMI,
                         LMI_NAH2PO4, 1)
    NA_MI_NAH2PO4 = NAH2PO4[0]
    H2PO4_MI_NAH2PO4 = NAH2PO4[1]
    CLHCO3 = CLHCO3_MI(CM_CL, CI_CL, CM_HCO3, CI_HCO3, Z_CL, Z_HCO3, VM, VI, AMI,
                       LMI_CLHCO3, 1)
    CL_MI_CLHCO3 = CLHCO3[0]
    HCO3_MI_CLHCO3 = CLHCO3[1]
    CLHCO2 = CLHCO2_MI(CM_CL, CI_CL, CM_HCO2, CI_HCO2, Z_CL, Z_HCO2, VM, VI, AMI,
                       LMI_CLHCO2, 1)
    CL_MI_CLHCO2 = CLHCO2[0]
    HCO2_MI_CLHCO2 = CLHCO2[1]
    NAHCO3 = NAHCO3_IS(CI_NA, CS_NA, CI_HCO3, CS_HCO3, Z_NA, Z_HCO3, VI, VS, AIS,
                       LIS_NAHCO3, 1)
    NA_IS_NAHCO3 = NAHCO3[0]
    HCO3_IS_NAHCO3 = NAHCO3[1]
    KCL = KCL_IS(CI_K, CS_K, CI_CL, CS_CL, Z_K, Z_CL, VI, VS, AIS, LIS_KCL, 1)
    K_IS_KCL = KCL[0]
    CL_IS_KCL = KCL[1]
    NA_CLHCO3 = NA_CLHCO3_IS(CI_NA, CS_NA, CI_CL, CS_CL, CI_HCO3, CS_HCO3, Z_NA, Z_CL, Z_HCO3,
                             VI, VS, AIS, LIS_NA_CLHCO3, 1)
    NA_IS_NA_CLHCO3 = NA_CLHCO3[0]
    CL_IS_NA_CLHCO3 = NA_CLHCO3[1]
    HCO3_IS_NA_CLHCO3 = NA_CLHCO3[2]
    # THE NAH EXCHANGER TRANSLATE CONCENTRATIONS TO THE NAH MODEL
    MYNAH = NAH(CI_H, CI_NA, CI_NH4, CM_H, CM_NA, CM_NH4, 1)
    JNAH_NA = MYNAH[0]
    JNAH_H = MYNAH[1]
    JNAH_NH4 = MYNAH[2]
    JNHE3_NA = NNHE3 * AMI * JNAH_NA
    JNHE3_H = NNHE3 * AMI * JNAH_H
    JNHE3_NH4 = NNHE3 * AMI * JNAH_NH4

    FIKM_NA = FIKM_NA + NA_MI_NAGLUC + NA_MI_NAH2PO4 + JNHE3_NA
    FIKM_CL = FIKM_CL + CL_MI_CLHCO2 + CL_MI_CLHCO3
    FIKM_HCO3 = FIKM_HCO3 + HCO3_MI_CLHCO3
    FIKM_H2PO4 = FIKM_H2PO4 + H2PO4_MI_NAH2PO4
    FIKM_HCO2 = FIKM_HCO2 + HCO2_MI_CLHCO2
    FIKM_GLUC = FIKM_GLUC + GLUC_MI_NAGLUC
    FIKM_H = FIKM_H + JNHE3_H
    FIKM_NH4 = FIKM_NH4 + JNHE3_NH4

    FIKS_NA = FIKS_NA + NA_IS_NAHCO3 + NA_IS_NA_CLHCO3
    FIKS_K = FIKS_K + K_IS_KCL
    FIKS_CL = FIKS_CL + CL_IS_KCL + CL_IS_NA_CLHCO3
    FIKS_HCO3 = FIKS_HCO3 + HCO3_IS_NA_CLHCO3 + HCO3_IS_NAHCO3

    JK_NA = JK_NA + NA_IS_NAHCO3 + NA_IS_NA_CLHCO3
    JK_K = JK_K + K_IS_KCL
    JK_CL = JK_CL + CL_IS_KCL + CL_IS_NA_CLHCO3
    JK_HCO3 = JK_HCO3 + HCO3_IS_NA_CLHCO3 + HCO3_IS_NAHCO3


    # SODIUM PUMPS
    NAK = NAK_ATP(CI_K, CS_K, CI_NA, CS_NA, CE_K, CE_NH4, 1)
    ATIS_NA = NAK[0]
    ATIS_K = NAK[1]
    ATIS_NH4 = NAK[2]
    ATMI_H = AT_MI_H(CM_H, CI_H, VM, VI, Z_H, 1)

    JK_NA = JK_NA + AIE * ATIS_NA
    JK_K = JK_K + AIE * ATIS_K
    JK_NH4 = JK_NH4 + AIE * ATIS_NH4
    FIKS_NA = FIKS_NA + AIS * ATIS_NA
    FIKS_K = FIKS_K + AIS * ATIS_K
    FIKS_NH4 = FIKS_NH4 + AIS * ATIS_NH4
    # PROTON PUMPS
    FIKM_H = FIKM_H + AMI * ATMI_H
#ESTABLISH THE ERROR VECTORS, THE "phi" ARRAY.

    #FIRST FOR THE INTERSPACE  ELECTRONEUTRALITY

    phiE_EN =  Z_NA * CE_NA + Z_K * CE_K + Z_CL * CE_CL + Z_HCO3 * CE_HCO3 + Z_H2CO3 * CE_H2CO3 \
                 + Z_CO2 * CE_CO2 + Z_HPO4 * CE_HPO4 + Z_H2PO4 * CE_H2PO4 + Z_UREA * CE_UREA + Z_NH3 * CE_NH3 \
                 + Z_NH4 * CE_NH4 + Z_H * CE_H + Z_HCO2 * CE_HCO2 + Z_H2CO2 * CE_H2CO2 + Z_GLUC * CE_GLUC
    if 0:
    # MASS CONSERVATION IN THE TIME-DEPENDENT CASE
        phiE_VLM = FEVS - FEVM + RTAU * (CHVL - CHVL)
        QE_NA = FEKS_NA - FEKM_NA + RTAU * (CE_NA * CHVL - CE_NA * CHVL)
        QE_K = FEKS_K - FEKM_K + RTAU * (CE_K * CHVL - CE_K * CHVL)
        QE_CL = FEKS_CL - FEKM_CL + RTAU * (CE_CL * CHVL - CE_CL * CHVL)
        QE_HCO3 = FEKS_HCO3 - FEKM_HCO3 + RTAU * (CE_HCO3 * CHVL - CE_HCO3 * CHVL)
        QE_H2CO3 = FEKS_H2CO3 - FEKM_H2CO3 + RTAU * (CE_H2CO3 * CHVL - CE_H2CO3 * CHVL)
        QE_CO2 = FEKS_CO2 - FEKM_CO2 + RTAU * (CE_CO2 * CHVL - CE_CO2 * CHVL)
        QE_HPO4 = FEKS_HPO4 - FEKM_HPO4 + RTAU * (CE_HPO4 * CHVL - CE_HPO4 * CHVL)
        QE_H2PO4 = FEKS_H2PO4 - FEKM_H2PO4 + RTAU * (CE_H2PO4 * CHVL - CE_H2PO4 * CHVL)
        QE_UREA = FEKS_UREA - FEKM_UREA + RTAU * (CE_UREA * CHVL - CE_UREA * CHVL)
        QE_NH3 = FEKS_NH3 - FEKM_NH3 + RTAU * (CE_NH3 * CHVL - CE_NH3 * CHVL)
        QE_NH4 = FEKS_NH4 - FEKM_NH4 + RTAU * (CE_NH4 * CHVL - CE_NH4 * CHVL)
        QE_H = FEKS_H - FEKM_H + RTAU * (CE_H * CHVL - CE_H * CHVL)
        QE_HCO2 = FEKS_HCO2 - FEKM_HCO2 + RTAU * (CE_HCO2 * CHVL - CE_HCO2 * CHVL)
        QE_H2CO2 = FEKS_H2CO2 - FEKM_H2CO2 + RTAU * (CE_H2CO2 * CHVL - CE_H2CO2 * CHVL)
        QE_GLUC = FEKS_GLUC - FEKM_GLUC + RTAU * (CE_GLUC * CHVL - CE_GLUC * CHVL)

        phiE_VLM = phiE_VLM - JV
        QE_NA = QE_NA - JK_NA
        QE_K = QE_K - JK_K
        QE_CL = QE_CL - JK_CL
        QE_HCO3 = QE_HCO3 - JK_HCO3
        QE_H2CO3 = QE_H2CO3 - JK_H2CO3
        QE_CO2 = QE_CO2 - JK_CO2
        QE_HPO4 = QE_HPO4 - JK_HPO4
        QE_H2PO4 = QE_H2PO4 - JK_H2PO4
        QE_UREA = QE_UREA - JK_UREA
        QE_NH3 = QE_NH3 - JK_NH3
        QE_NH4 = QE_NH4 - JK_NH4
        QE_H = QE_H - JK_H
        QE_HCO2 = QE_HCO2 - JK_HCO2
        QE_H2CO2 = QE_H2CO2 - JK_H2CO2
        QE_GLUC = QE_GLUC - JK_GLUC

    else:
        # MASS  CONSERVATION IN THE STEADY - STATE CASE
        phiE_VLM = FEVS - FEVM
        QE_NA = FEKS_NA - FEKM_NA
        QE_K = FEKS_K - FEKM_K
        QE_CL = FEKS_CL - FEKM_CL
        QE_HCO3 = FEKS_HCO3 - FEKM_HCO3
        QE_H2CO3 = FEKS_H2CO3 - FEKM_H2CO3
        QE_CO2 = FEKS_CO2 - FEKM_CO2
        QE_HPO4 = FEKS_HPO4 - FEKM_HPO4
        QE_H2PO4 = FEKS_H2PO4 - FEKM_H2PO4
        QE_UREA = FEKS_UREA - FEKM_UREA
        QE_NH3 = FEKS_NH3 - FEKM_NH3
        QE_NH4 = FEKS_NH4 - FEKM_NH4
        QE_H = FEKS_H - FEKM_H
        QE_HCO2 = FEKS_HCO2 - FEKM_HCO2
        QE_H2CO2 = FEKS_H2CO2 - FEKM_H2CO2
        QE_GLUC = FEKS_GLUC - FEKM_GLUC

        phiE_VLM = phiE_VLM - JV
        QE_NA = QE_NA - JK_NA
        QE_K = QE_K - JK_K
        QE_CL = QE_CL - JK_CL
        QE_HCO3 = QE_HCO3 - JK_HCO3
        QE_H2CO3 = QE_H2CO3 - JK_H2CO3
        QE_CO2 = QE_CO2 - JK_CO2
        QE_HPO4 = QE_HPO4 - JK_HPO4
        QE_H2PO4 = QE_H2PO4 - JK_H2PO4
        QE_UREA = QE_UREA - JK_UREA
        QE_NH3 = QE_NH3 - JK_NH3
        QE_NH4 = QE_NH4 - JK_NH4
        QE_H = QE_H - JK_H
        QE_HCO2 = QE_HCO2 - JK_HCO2
        QE_H2CO2 = QE_H2CO2 - JK_H2CO2
        QE_GLUC = QE_GLUC - JK_GLUC
    # THEN FOR THE CELLS  ELECTRONEUTRALITY
    phiI_EN = IMP * ZIMP - CBUF
    phiI_EN = phiI_EN + Z_NA * CI_NA + Z_K * CI_K \
                    + Z_CL * CI_CL + Z_HCO3 * CI_HCO3 + Z_H2CO3 * CI_H2CO3 + Z_CO2 * CI_CO2 \
                    + Z_HPO4 * CI_HPO4 + Z_H2PO4 * CI_H2PO4 + Z_UREA * CI_UREA \
                    + Z_NH3 * CI_NH3 + Z_NH4 * CI_NH4 + Z_H * CI_H \
                    + Z_HCO2 * CI_HCO2 + Z_H2CO2 * CI_H2CO2 + Z_GLUC * CI_GLUC
    if 0:
        # MASS CONSERVATION IN THE TIME - DEPENDENT CASE
        phiI_VLM = FIVS - FIVM + JV + RTAU * (CLVL - CLVL)

        QI_NA = FIKS_NA - FIKM_NA + JK_NA + RTAU * (
                    CI_NA * CLVL - CI_NA * CLVL)
        QI_K = FIKS_K - FIKM_K + JK_K + RTAU * (
                    CI_K * CLVL - CI_K * CLVL)
        QI_CL = FIKS_CL - FIKM_CL + JK_CL + RTAU * (
                    CI_CL * CLVL - CI_CL * CLVL)
        QI_HCO3 = FIKS_HCO3 - FIKM_HCO3 + JK_HCO3 + RTAU * (
                    CI_HCO3 * CLVL - CI_HCO3 * CLVL)
        QI_H2CO3 = FIKS_H2CO3 - FIKM_H2CO3 + JK_H2CO3 + RTAU * (
                    CI_H2CO3 * CLVL - CI_H2CO3 * CLVL)
        QI_CO2 = FIKS_CO2 - FIKM_CO2 + JK_CO2 + RTAU * (
                    CI_CO2 * CLVL - CI_CO2 * CLVL)
        QI_HPO4 = FIKS_HPO4 - FIKM_HPO4 + JK_HPO4 + RTAU * (
                    CI_HPO4 * CLVL - CI_HPO4 * CLVL)
        QI_H2PO4 = FIKS_H2PO4 - FIKM_H2PO4 + JK_H2PO4 + RTAU * (
                    CI_HPO4 * CLVL - CI_HPO4 * CLVL)
        QI_UREA = FIKS_UREA - FIKM_UREA + JK_UREA + RTAU * (
                    CI_UREA * CLVL - CI_UREA * CLVL)
        QI_NH3 = FIKS_NH3 - FIKM_NH3 + JK_NH3 + RTAU * (
                    CI_NH3 * CLVL - CI_NH3 * CLVL)
        QI_NH4 = FIKS_NH4 - FIKM_NH4 + JK_NH4 + RTAU * (
                    CI_NH4 * CLVL - CI_NH4 * CLVL)
        QI_H = FIKS_H - FIKM_H + JK_H + RTAU * (
                    CI_H * CLVL - CI_H * CLVL)
        QI_HCO2 = FIKS_HCO2 - FIKM_HCO2 + JK_HCO2 + RTAU * (
                    CI_HCO2 * CLVL - CI_HCO2 * CLVL)
        QI_H2CO2 = FIKS_H2CO2 - FIKM_H2CO2 + JK_H2CO2 + RTAU * (
                    CI_H2CO2 * CLVL - CI_H2CO2 * CLVL)
        QI_GLUC = FIKS_GLUC - FIKM_GLUC + JK_GLUC + RTAU * (
                    CI_GLUC * CLVL - CI_GLUC * CLVL)

        # THE PROTON FLUX MUST INCLUDE THE CELLULAR BUFFERS
        QI_H = QI_H - RTAU * (
                CBUF * CLVL - CBUF * CLVL)
    else:
        # MASS CONSERVATION IN THE STEADY-STATE CASE
        phiI_VLM = FIVS - FIVM + JV
        QI_NA = FIKS_NA - FIKM_NA + JK_NA
        QI_K = FIKS_K - FIKM_K + JK_K
        QI_CL = FIKS_CL - FIKM_CL + JK_CL
        QI_HCO3 = FIKS_HCO3 - FIKM_HCO3 + JK_HCO3
        QI_H2CO3 = FIKS_H2CO3 - FIKM_H2CO3 + JK_H2CO3
        QI_CO2 = FIKS_CO2 - FIKM_CO2 + JK_CO2
        QI_HPO4 = FIKS_HPO4 - FIKM_HPO4 + JK_HPO4
        QI_H2PO4 = FIKS_H2PO4 - FIKM_H2PO4 + JK_H2PO4
        QI_UREA = FIKS_UREA - FIKM_UREA + JK_UREA
        QI_NH3 = FIKS_NH3 - FIKM_NH3 + JK_NH3
        QI_NH4 = FIKS_NH4 - FIKM_NH4 + JK_NH4
        QI_H = FIKS_H - FIKM_H + JK_H
        QI_HCO2 = FIKS_HCO2 - FIKM_HCO2 + JK_HCO2
        QI_H2CO2 = FIKS_H2CO2 - FIKM_H2CO2 + JK_H2CO2
        QI_GLUC = FIKS_GLUC - FIKM_GLUC + JK_GLUC

    phiE_VLM = phiScale(phiE_VLM, Scale)
    QE_NA = phiScale(QE_NA, Scale)
    QE_K = phiScale(QE_K, Scale)
    QE_CL = phiScale(QE_CL, Scale)
    QE_HCO3 = phiScale(QE_HCO3, Scale)
    QE_H2CO3 = phiScale(QE_H2CO3, Scale)
    QE_CO2 = phiScale(QE_CO2, Scale)
    QE_HPO4 = phiScale(QE_HPO4, Scale)
    QE_H2PO4 = phiScale(QE_H2PO4, Scale)
    QE_UREA = phiScale(QE_UREA, Scale)
    QE_NH3 = phiScale(QE_NH3, Scale)
    QE_NH4 = phiScale(QE_NH4, Scale)
    QE_H = phiScale(QE_H, Scale)
    QE_HCO2 = phiScale(QE_HCO2, Scale)
    QE_H2CO2 = phiScale(QE_H2CO2, Scale)
    QE_GLUC = phiScale(QE_GLUC, Scale)

    phiI_VLM = phiScale(phiI_VLM, Scale)
    QI_NA = phiScale(QI_NA, Scale)
    QI_K = phiScale(QI_K, Scale)
    QI_CL = phiScale(QI_CL, Scale)
    QI_HCO3 = phiScale(QI_HCO3, Scale)
    QI_H2CO3 = phiScale(QI_H2CO3, Scale)
    QI_CO2 = phiScale(QI_CO2, Scale)
    QI_HPO4 = phiScale(QI_HPO4, Scale)
    QI_H2PO4 = phiScale(QI_H2PO4, Scale)
    QI_UREA = phiScale(QI_UREA, Scale)
    QI_NH3 = phiScale(QI_NH3, Scale)
    QI_NH4 = phiScale(QI_NH4, Scale)
    QI_H = phiScale(QI_H, Scale)
    QI_HCO2 = phiScale(QI_HCO2, Scale)
    QI_H2CO2 = phiScale(QI_H2CO2, Scale)
    QI_GLUC = phiScale(QI_GLUC, Scale)

    # NOW SET THE phiS IN TERMS OF THE SOLUTE GENERATION RATES
    # FIRST THE NON-REACTIVE SPECIES
    phi[-1+1] = phiE_EN
    phi[-1+2] = phiE_VLM

    phiE_NA = QE_NA
    phi[-1+3] = phiE_NA
    phiE_K = QE_K
    phi[-1+4] = phiE_K
    phiE_CL = QE_CL
    phi[-1+5] = phiE_CL
    phiE_UREA = QE_UREA
    phi[-1+11] = phiE_UREA
    phiE_GLUC = QE_GLUC
    phi[-1+16] = phiE_GLUC

    SOLS = 15
    kk = SOLS + 1
    phi[-1+1 + kk] = phiI_EN
    phi[-1+2 + kk] = phiI_VLM
    phiI_NA = QI_NA
    phi[-1+3+kk] = phiI_NA
    phiI_K = QI_K
    phi[-1+4+kk] = phiI_K
    phiI_CL = QI_CL
    phi[-1+5+kk] = phiI_CL
    phiI_UREA = QI_UREA
    phi[-1+11+kk] = phiI_UREA
    phiI_GLUC = QI_GLUC
    phi[-1+16+kk] = phiI_GLUC

    # CO2, FORMATE, PHOSPHATE, AND AMMONIA CONTENT:
    phi[-1+8] = QE_HCO3 + QE_H2CO3 + QE_CO2
    phi[-1+9] = QE_HPO4 + QE_H2PO4
    phi[-1+12] = QE_NH3 + QE_NH4
    phi[-1+14] = QE_HCO2 + QE_H2CO2

    phi[-1+8+kk] = QI_HCO3 + QI_H2CO3 + QI_CO2
    phi[-1+9+kk] = QI_HPO4 + QI_H2PO4
    phi[-1+12+kk] = QI_NH3 + QI_NH4- 1e6*QIAMM
    phi[-1+14+kk] = QI_HCO2 + QI_H2CO2
    # FORMATE, PHOSPHATE, AND AMMMONIA PH EQUILIBRIUM:
    phiE_H2PO4 = EBUF(CE_H, PKP, CE_HPO4, CE_H2PO4)
    phi[-1+10] = phiE_H2PO4
    phiE_NH4 = EBUF(CE_H, PKN, CE_NH3, CE_NH4)
    phi[-1+13] = phiE_NH4
    phiE_H2CO2 = EBUF(CE_H, PKF, CE_HCO2, CE_H2CO2)
    phi[-1+15] = phiE_H2CO2
    # phiE_H2CO3 = EBUF(CE_H, PKC, CE_HCO3, CE_H2CO3)
    phiI_H2PO4 = EBUF(CI_H, PKP, CI_HPO4, CI_H2PO4)
    phi[-1+10+kk] = phiI_H2PO4
    phiI_NH4 = EBUF(CI_H, PKN, CI_NH3, CI_NH4)
    phi[-1+13+kk] = phiI_NH4
    phiI_H2CO2 = EBUF(CI_H, PKF, CI_HCO2, CI_H2CO2)
    phi[-1+15+kk] = phiI_H2CO2
    # HYDRATION AND DHYDRATION OF CO2
    phiE_CO2 = QE_CO2 + 1.e6*CHVL*(KHY_4*CE_CO2 - KDHY_4*CE_H2CO3)
    phi[-1+7] = phiE_CO2
    phiI_CO2 = QI_CO2 + 1.e6 * CLVL * (KHY * CI_CO2 - KDHY * CI_H2CO3)
    phi[-1+7+kk] = phiI_CO2
    # THE ERROR TERM FOR BICARBONATE GENERATION IS REPLACED BY CONSERVATION
    # OF CHARGE IN THE BUFFER REACTIONS.
    phiE_H2CO3 = - QE_HCO3 + QE_H2PO4 + QE_NH4 + QE_H + QE_H2CO2
    phi[-1+6] = phiE_H2CO3
    phiI_H2CO3 = - QI_HCO3 + QI_H2PO4 + QI_NH4 + QI_H + QI_H2CO2
    phi[-1+6+kk] = phiI_H2CO3

    # CELL BUFFER CONTENT AND PH EQUILIBRIUM
    phi[-1+3+2*SOLS] = CBUF + HCBUF -(TBUF*CLVL0/CLVL)
    C_I_H2PO4 = 0 if (CI_H2PO4 * CBUF == 0 or ((CI_HPO4 * HCBUF) / (CI_H2PO4 * CBUF)) <= 0) else math.log10((CI_HPO4 * HCBUF) / (CI_H2PO4 * CBUF))
    phi[-1+4+2*SOLS] = PKB - PKP - C_I_H2PO4
    # THE OPEN CIRCUIT CONDITION
    # KK=34
    CURE = F * (
            Z_NA * FEKM_NA + Z_K * FEKM_K + Z_CL * FEKM_CL + Z_HCO3 *
            FEKM_HCO3 +
            Z_H2CO3 * FEKM_H2CO3 + Z_CO2 * FEKM_CO2 + Z_HPO4 * FEKM_HPO4 + Z_H2PO4 * FEKM_H2PO4 +
            Z_UREA * FEKM_UREA + Z_NH3 * FEKM_NH3 + Z_NH4 * FEKM_NH4 + Z_H * FEKM_H + Z_HCO2 * FEKM_HCO2 +
            Z_H2CO2 * FEKM_H2CO2 + Z_GLUC * FEKM_GLUC)
    phi_CUR = CURE
    CURI = F * (
            Z_NA * FIKM_NA + Z_K * FIKM_K + Z_CL * FIKM_CL + Z_HCO3 *
            FIKM_HCO3 +
            Z_H2CO3 * FIKM_H2CO3 + Z_CO2 * FIKM_CO2 + Z_HPO4 * FIKM_HPO4 + Z_H2PO4 * FIKM_H2PO4 +
            Z_UREA * FIKM_UREA + Z_NH3 * FIKM_NH3 + Z_NH4 * FIKM_NH4 + Z_H * FIKM_H + Z_HCO2 * FIKM_HCO2 +
            Z_H2CO2 * FIKM_H2CO2 + Z_GLUC * FIKM_GLUC)

    phi_CUR = phi_CUR + CURI
    phi[4 + 2 * 15] = phi_CUR
    return phi[i]
import sympy as sympy
from scipy.misc import derivative
from operator import add
import numpy as np

from sympy import *
from sympy import symbols, diff

        # def f(x, y, z):
        #     return x * 3 + x ** 2 + z * y


def partial_derivative(func, f_num, var=0, point=[]):
    args = point[:]

    def wraps(x):
        args[var] = x
        # print(args)
        return func(f_num, args)

    return derivative(wraps, point[var], dx=1e-6)


#for T in range(0):
    # print('x: %d t: %d'%(x,t))
    #print(AE[x])
    #print(AE)
GUESS = [VE, PE, CE_NA, CE_K,
         CE_CL, CE_HCO3, CE_H2CO3, CE_CO2,
         CE_HPO4, CE_H2PO4, CE_UREA, CE_NH3,
         CE_NH4, CE_HCO2, CE_H2CO2, CE_GLUC,
         VI, IMP, CI_NA, CI_K,
         CI_CL, CI_HCO3, CI_H2CO3, CI_CO2,
         CI_HPO4, CI_H2PO4, CI_UREA, CI_NH3,
         CI_NH4, CI_HCO2, CI_H2CO2, CI_GLUC,
         CBUF, HCBUF, VM]

# phi_res =[None for i in range(35)]
phi_res = {}
# dif = {}
dif = [[None for i in range(35) ]   for j in range(35) ]
for i in range(0,35):
    phi[i] = EQS(i, GUESS)
    phi_res[i] = phi[i]
    for j in range(0,35):

        dif[i][j] = partial_derivative(EQS, i, j, GUESS)
        # print('partial_derivative=', dif, 'i=', i, 'j=', j)


print('dif=', dif)
print('phi=', phi_res)



# def linearsolver(A,b):
#   n = len(A)
#   M = A
#
#   i = 0
#   for x in M:
#    x.append(b[i])
#    i += 1
#
#   for k in range(n):
#    for i in range(k,n):
#      if abs(M[i][k]) > abs(M[k][k]):
#         M[k], M[i] = M[i],M[k]
#      else:
#         pass
#
#    for j in range(k+1,n):
#        q = float(M[j][k]) / M[k][k]
#        for m in range(k, n+1):
#           M[j][m] -=  q * M[k][m]
#
#   x = [0 for i in range(n)]
#
#   x[n-1] =float(M[n-1][n])/M[n-1][n-1]
#   for i in range (n-1,-1,-1):
#     z = 0
#     for j in range(i+1,n):
#         z = z  + float(M[i][j])*x[j]
#     x[i] = float(M[i][n] - z)/M[i][i]
#   print(x)
#   print(phi)

#   print(linearsolver(dif, phi))
