import math
import numpy as np
from scipy.optimize import *
import sys

from PCT_GLOB import *

sys.path.insert(0, '../PCT')
from PCT_GLOB import *

np.log(sys.float_info.min)
np.log(sys.float_info.min * sys.float_info.epsilon)


def LMMSC(C_a, C_b):
    # Logarithm mean membrane solute concentration
    import math
    math.log(sys.float_info.min)
    math.log(sys.float_info.min * sys.float_info.epsilon)
    if C_a / C_b > 0 and C_a - C_b != 0:
        return (C_a - C_b) / (math.log10(C_a / C_b))
    else:
        return C_b


def NAH(CI_H, CI_NA, CI_NH4, CM_H, CM_NA, CM_NH4, param_NAH):
    if param_NAH == 0:
        return [ 0, 0, 0 ]
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
        return [ JNAH_NA, JNAH_H, JNAH_NH4 ]



def KCL_IS(CI_K, CS_K, CI_CL, CS_CL, Z_K, Z_CL, VI, VS, AIS, LIS_KCL, param_KCL_IS):
    if param_KCL_IS == 0:
        return [ 0, 0 ]
    else:
        XI_K = EPS(CI_K, Z_K, VI)
        XI_CL = EPS(CI_CL, Z_CL, VI)
        XS_K = EPS(CS_K, Z_K, VS)
        XS_CL = EPS(CS_CL, Z_CL, VS)
        K_IS_KCL = LIS_KCL * AIS * (XI_K - XS_K + XI_CL - XS_CL)
        CL_IS_KCL = LIS_KCL * AIS * (XI_K - XS_K + XI_CL - XS_CL)
    return [ K_IS_KCL, CL_IS_KCL ]


def EPS(C, Z, V):
    if C > 0 and C != 0:
        return RTE * math.log10(C) + Z * F * V * 1.e-6
    else:
        return Z * F * V * 1.e-6


def SGLT_MI(CM_NA, CI_NA, CM_GLUC, CI_GLUC, Z_GLUC, Z_NA, VM, VI, AMI, LMI_NAGLUC, param_SGLT_MI):
    if param_SGLT_MI == 0:
        return [ 0, 0 ]
    else:
        XM_NA = EPS(CM_NA, Z_NA, VM)
        XM_GLUC = EPS(CM_GLUC, Z_GLUC, VM)
        XI_NA = EPS(CI_NA, Z_NA, VI)
        XI_GLUC = EPS(CI_GLUC, Z_GLUC, VI)
        NA_MI_NAGLUC = LMI_NAGLUC * AMI * (XM_NA - XI_NA + XM_GLUC - XI_GLUC)
        GLUC_MI_NAGLUC = LMI_NAGLUC * AMI * (XM_NA - XI_NA + XM_GLUC - XI_GLUC)
        return [ NA_MI_NAGLUC, GLUC_MI_NAGLUC ]


def NAK_ATP(CI_K, CS_K, CI_NA, CS_NA, CE_K, CE_NH4, param_NAK_ATP):
    if param_NAK_ATP == 0:
        return [ 0, 0, 0 ]
    else:
        KNPN = 0.0002 * (1.0 + CI_K / .00833)
        # SODIUM AFFINITY
        KNPK = 0.0001 * (1.0 + CS_NA / .0185)
        # ATPASE TRANSPORTER FLUX IN IS MEMBRANE
        ATIS_NA = NP * (CI_NA / (KNPN + CI_NA)) ** 3 * (CS_K / (KNPK + CS_K)) ** 2
        # ALLOW FOR COMPETITION BETWEEN K+ AND NH4+

        ATIS_K = -ATIS_NA * 0.667 * CE_K / (CE_K + CE_NH4 / KNH4)
        ATIS_NH4 = -ATIS_NA * 0.667 * CE_NH4 / (KNH4 * CE_K + CE_NH4)
        return [ ATIS_NA, ATIS_K, ATIS_NH4 ]


def GOLDMAN(hab, A, Zab, Va, Vb, ca, cb, param_GOLDMAN):
    Zab = (Zab * F * (Va - Vb) * 1.e-6) / RTE

    if param_GOLDMAN == 0:
        return [ 0 ]
    elif Zab == 0 or Va == Vb:
        return hab * A * (ca - cb)
    elif Zab > 0:
        return hab * A * Zab * (ca - cb * math.exp(-Zab)) / (1 - math.exp(-Zab))
    else:
        return hab * A * Zab * (ca * math.exp(Zab) - cb) / (math.exp(Zab) - 1)
        # normalized electrical potential difference


def LMMS(C_a, C_b):
    # Logarithm mean membrane solute concentration
    import math
    math.log(sys.float_info.min)
    math.log(sys.float_info.min * sys.float_info.epsilon)
    if C_a / C_b > 0 and C_a - C_b != 0:
        return (C_a - C_b) / (math.log10(C_a / C_b))
    else:
        return C_b


def EPS(C, Z, V):
    # Electrochemical Potential of Species i
    import math
    math.log(sys.float_info.min)
    math.log(sys.float_info.min * sys.float_info.epsilon)
    if C > 0 and C != 0:
        return RTE * math.log10(C) + Z * F * V * 1.e-6
    else:
        return Z * F * V * 1.e-6


def Zab(Z, Va, Vb):
    return Z * F * (Va - Vb) * 1.e-6 / RTE


def PHIScale(PHI, Scale):
    return PHI * Scale


def EQS(GUESS):
    # COMPUTE THE RELEVANT VOLUMES
    VE [ t ] = GUESS [ 0 ]
    PE [ t ] = GUESS [ 1 ]
    CE_NA [ t ] = GUESS [ 2 ]
    CE_K [ t ] = GUESS [ 3 ]
    CE_CL [ t ] = GUESS [ 4 ]
    CE_UREA [ t ] = GUESS [5 ]
    CE_GLUC [ t ] = GUESS [6 ]

    VI [ t ] = GUESS [7 ]
    IMP[t] = GUESS [8 ]
    CI_NA [ t ] = GUESS [9 ]
    CI_K [ t ] = GUESS [ 10 ]
    CI_CL [ t ] = GUESS [ 11 ]
    CI_UREA [ t ] = GUESS [ 12]
    CI_GLUC [ t ] = GUESS [ 13 ]
    VM [ t ] = GUESS [14]


    AE [ t ] = AE0 * ( 1 + MUA * (PE [ t ] - PM))
    AE [ t ] = AE0 if AE [ t ] < AE0 else AE [ t ]

    CHVL [ t ] = CHVL0 * (1.0 + MUV * (PE [ t ] - PM))
    CHVL [ t ] = CHVL0 if CHVL [ t ] < CHVL0 else CHVL [ t ]
    L [ t ] = CHVL [ t ]


    CLVL [ t ] = CLVL0 * IMP0 / IMP[t]
    CLVL [ t ] = CLVL0 if IMP == 0 else CLVL [ t ]
    L [ t ] = L [ t ] + CLVL [ t ]
    PI = PM

    # COMPUTE THE MEMBRANE FLUXES
    FEVM [ t ] = AME * LPME * (PM - PE [ t ] - RT * IMPM) / RT
    #print('PE',PE[t])
    FEVS [ t ] = AE [ t ] * LPES * (RT * IMPS + PE [ t ] - PS) / RT
    
    FIVM [ t ] = AMI * LPMI * (RT * IMP[t] - RT * IMPM + PM - PI ) / RT
    
    FIVS [ t ] = AIS * LPIS * (PI - PS + RT * IMPS - RT * IMP[t]) / RT
    
    JV [ t ] = LPIS * AIE * (PI - PE [ t ] - RT * IMP[t]) / RT

    FEVM [ t ] = FEVM [ t ] + AME * LPME * (
            SME_NA * (CE_NA [ t ] - CM_NA) + SME_K * (
            CE_K [ t ] - CM_K) + SME_CL * (
                    CE_CL [ t ] - CM_CL) + SME_GLUC * (
                    CE_GLUC [ t ] - CM_GLUC) +  SME_UREA * (
                    CE_UREA [ t ] - CM_UREA))

    FEVS [ t ] = FEVS [ t ] + AE [ t ] * LPES * (
            SES_NA * (CS_NA - CE_NA [ t ]) + SES_K * (
            CS_K - CE_K [ t ]) + SES_CL * (
                    CS_CL - CE_CL [ t ]) + SES_UREA * (
                    CS_UREA - CE_UREA [ t ]) + SES_GLUC * (
                    CS_GLUC - CE_GLUC [ t ]) )

    FIVM [ t ] = FIVM [ t ] + AMI * LPMI * (
            SMI_NA * (CI_NA [ t ] - CM_NA) + SMI_K * (
            CI_K [ t ] - CM_K) + SMI_CL * (
                    CI_CL [ t ] - CM_CL) + SMI_GLUC * (
                    CI_GLUC [ t ] - CM_GLUC)+ SMI_UREA * (
                    CI_UREA [ t ] - CM_UREA) )

    FIVS [ t ] = FIVS [ t ] + AIS * LPIS * (
            SIS_NA * (CS_NA - CI_NA [ t ]) + SIS_K * (
            CS_K - CI_K [ t ]) + SIS_CL * (
                    CS_CL - CI_CL [ t ])  + SIS_GLUC * (
                    CS_GLUC - CI_GLUC [ t ]) +  SIS_UREA * (
                    CS_UREA - CI_UREA [ t ]))

    JV [ t ] = JV [ t ] + LPIS * AIE * (
            SIS_NA * (CE_NA [ t ] - CI_NA [ t ]) + SIS_K * (
            CE_K [ t ] - CI_K [ t ]) + SIS_CL * (
                    CE_CL [ t ] - CI_CL [ t ]) + SIS_UREA* (
                    CE_UREA [ t ] - CI_UREA [ t ]) + SIS_GLUC * (
                    CE_GLUC [ t ] - CI_GLUC [ t ]))

    CME_NA = LMMS(CE_NA [ t ], CM_NA)
    CME_K = LMMS(CE_K [ t ], CM_K)
    CME_CL = LMMS(CE_CL [ t ], CM_CL)
    CME_UREA = LMMS(CE_UREA [ t ], CM_UREA)
    # CME_NH3 = LMMS(CE_NH3 [ t ], CM_NH3)
    # CME_NH4 = LMMS(CE_NH4 [ t ], CM_NH4)
    # CME_H = LMMS(CE_H [ t ], CM_H)
    # CME_HCO2 = LMMS(CE_HCO2 [ t ], CM_HCO2)
    # CME_H2CO2 = LMMS(CE_H2CO2 [ t ], CM_H2CO2)
    CME_GLUC = LMMS(CE_GLUC [ t ], CM_GLUC)

    CES_NA = LMMSC(CE_NA [ t ], CS_NA)
    CES_K = LMMSC(CE_K [ t ], CS_K)
    CES_CL = LMMSC(CE_CL [ t ], CS_CL)
    # CES_HCO3 = LMMSC(CE_HCO3 [ t ], CS_HCO3)
    # CES_H2CO3 = LMMSC(CE_H2CO3 [ t ], CS_H2CO3)
    # CES_CO2 = LMMSC(CE_CO2 [ t ], CS_CO2)
    # CES_HPO4 = LMMSC(CE_HPO4 [ t ], CS_HPO4)
    # CES_H2PO4 = LMMSC(CE_H2PO4 [ t ], CS_H2PO4)
    CES_UREA = LMMSC(CE_UREA [ t ], CS_UREA)
    # CES_NH3 = LMMSC(CE_NH3 [ t ], CS_NH3)
    # CES_NH4 = LMMSC(CE_NH4 [ t ], CS_NH4)
    # CES_H = LMMSC(CE_H [ t ], CS_H)
    # CES_HCO2 = LMMSC(CE_HCO2 [ t ], CS_HCO2)
    # CES_H2CO2 = LMMSC(CE_H2CO2 [ t ], CS_H2CO2)
    CES_GLUC = LMMSC(CE_GLUC [ t ], CS_GLUC)

    CMI_NA = LMMSC(CI_NA [ t ], CM_NA)
    CMI_K = LMMSC(CI_K [ t ], CM_K)
    CMI_CL = LMMSC(CI_CL [ t ], CM_CL)
    # CMI_HCO3 = LMMSC(CI_HCO3 [ t ], CM_HCO3)
    # CMI_H2CO3 = LMMSC(CI_H2CO3 [ t ], CM_H2CO3)
    # CMI_CO2 = LMMSC(CI_CO2 [ t ], CM_CO2)
    # CMI_HPO4 = LMMSC(CI_HPO4 [ t ], CM_HPO4)
    # CMI_H2PO4 = LMMSC(CI_H2PO4 [ t ], CM_H2PO4)
    CMI_UREA = LMMSC(CI_UREA [ t ], CM_UREA)
    # CMI_NH3 = LMMSC(CI_NH3 [ t ], CM_NH3)
    # CMI_NH4 = LMMSC(CI_NH4 [ t ], CM_NH4)
    # CMI_H = LMMSC(CI_H [ t ], CM_H)
    # CMI_HCO2 = LMMSC(CI_HCO2 [ t ], CM_HCO2)
    # CMI_H2CO2 = LMMSC(CI_H2CO2 [ t ], CM_H2CO2)
    CMI_GLUC = LMMSC(CI_GLUC [ t ], CM_GLUC)

    CIS_NA = LMMSC(CI_NA [ t ], CS_NA)
    CIS_K = LMMSC(CI_K [ t ], CS_K)
    CIS_CL = LMMSC(CI_CL [ t ], CS_CL)
    # CIS_HCO3 = LMMSC(CI_HCO3 [ t ], CS_HCO3)
    # CIS_H2CO3 = LMMSC(CI_H2CO3 [ t ], CS_H2CO3)
    # CIS_CO2 = LMMSC(CI_CO2 [ t ], CS_CO2)
    # CIS_HPO4 = LMMSC(CI_HPO4 [ t ], CS_HPO4)
    # CIS_H2PO4 = LMMSC(CI_H2PO4 [ t ], CS_H2PO4)
    CIS_UREA = LMMSC(CI_UREA [ t ], CS_UREA)
    # CIS_NH3 = LMMSC(CI_NH3 [ t ], CS_NH3)
    # CIS_NH4 = LMMSC(CI_NH4 [ t ], CS_NH4)
    # CIS_H = LMMSC(CI_H [ t ], CS_H)
    # CIS_HCO2 = LMMSC(CI_HCO2 [ t ], CS_HCO2)
    # CIS_H2CO2 = LMMSC(CI_H2CO2 [ t ], CS_H2CO2)
    CIS_GLUC = LMMSC(CI_GLUC [ t ], CS_GLUC)

    CIE_NA = LMMSC(CI_NA [ t ], CE_NA [ t ])
    CIE_K = LMMSC(CI_K [ t ], CE_K [ t ])
    CIE_CL = LMMSC(CI_CL [ t ], CE_CL [ t ])
    # CIE_HCO3 = LMMSC(CI_HCO3 [ t ], CE_HCO3 [ t ])
    # CIE_H2CO3 = LMMSC(CI_H2CO3 [ t ], CE_H2CO3 [ t ])
    # CIE_CO2 = LMMSC(CI_CO2 [ t ], CE_CO2 [ t ])
    # CIE_HPO4 = LMMSC(CI_HPO4 [ t ], CE_HPO4 [ t ])
    # CIE_H2PO4 = LMMSC(CI_H2PO4 [ t ], CE_H2PO4 [ t ])
    CIE_UREA = LMMSC(CI_UREA [ t ], CE_UREA [ t ])
    # CIE_NH3 = LMMSC(CI_NH3 [ t ], CE_NH3 [ t ])
    # CIE_NH4 = LMMSC(CI_NH4 [ t ], CE_NH4 [ t ])
    # print('CE_H', CE_H [ t ])
    # print('CI_H', CI_H [ t ])
    # CIE_H = LMMSC(CI_H [ t ], CE_H [ t ])
    # CIE_HCO2 = LMMSC(CI_HCO2 [ t ], CE_HCO2 [ t ])
    # CIE_H2CO2 = LMMSC(CI_H2CO2 [ t ], CE_H2CO2 [ t ])
    CIE_GLUC = LMMSC(CI_GLUC [ t ], CE_GLUC [ t ])

    # CONVECTIVE FLUXES
    FEKM_NA [ t ] = FEVM [ t ] * (1.00 - SME_NA) * CME_NA
    FEKM_K [ t ] = FEVM [ t ] * (1.00 - SME_K) * CME_K
    FEKM_CL [ t ] = FEVM [ t ] * (1.00 - SME_CL) * CME_CL
    # FEKM_HCO3 [ t ] = FEVM [ t ] * (1.00 - SME_HCO3) * CME_HCO3
    # FEKM_H2CO3 [ t ] = FEVM [ t ] * (1.00 - SME_H2CO3) * CME_H2CO3
    # FEKM_CO2 [ t ] = FEVM [ t ] * (1.00 - SME_CO2) * CME_CO2
    # FEKM_HPO4 [ t ] = FEVM [ t ] * (1.00 - SME_HPO4) * CME_HPO4
    # FEKM_H2PO4 [ t ] = FEVM [ t ] * (1.00 - SME_H2PO4) * CME_H2PO4
    FEKM_UREA [ t ] = FEVM [ t ] * (1.00 - SME_UREA) * CME_UREA
    # FEKM_NH3 [ t ] = FEVM [ t ] * (1.00 - SME_NH3) * CME_NH3
    # FEKM_NH4 [ t ] = FEVM [ t ] * (1.00 - SME_NH4) * CME_NH4
    # FEKM_H [ t ] = FEVM [ t ] * (1.00 - SME_H) * CME_H
    # FEKM_HCO2 [ t ] = FEVM [ t ] * (1.00 - SME_HCO2) * CME_HCO2
    # FEKM_H2CO2 [ t ] = FEVM [ t ] * (1.00 - SME_H2CO2) * CME_H2CO2
    FEKM_GLUC [ t ] = FEVM [ t ] * (1.00 - SME_GLUC) * CME_GLUC

    FEKS_NA [ t ] = FEVS [ t ] * (1.00 - SES_NA) * CES_NA
    FEKS_K [ t ] = FEVS [ t ] * (1.00 - SES_K) * CES_K
    FEKS_CL [ t ] = FEVS [ t ] * (1.00 - SES_CL) * CES_CL
    # FEKS_HCO3 [ t ] = FEVS [ t ] * (1.00 - SES_HCO3) * CES_HCO3
    # FEKS_H2CO3 [ t ] = FEVS [ t ] * (1.00 - SES_H2CO3) * CES_H2CO3
    # FEKS_CO2 [ t ] = FEVS [ t ] * (1.00 - SES_CO2) * CES_CO2
    # FEKS_HPO4 [ t ] = FEVS [ t ] * (1.00 - SES_HPO4) * CES_HPO4
    # FEKS_H2PO4 [ t ] = FEVS [ t ] * (1.00 - SES_H2PO4) * CES_H2PO4
    FEKS_UREA [ t ] = FEVS [ t ] * (1.00 - SES_UREA) * CES_UREA
    # FEKS_NH3 [ t ] = FEVS [ t ] * (1.00 - SES_NH3) * CES_NH3
    # FEKS_NH4 [ t ] = FEVS [ t ] * (1.00 - SES_NH4) * CES_NH4
    # FEKS_H [ t ] = FEVS [ t ] * (1.00 - SES_H) * CES_H
    # FEKS_HCO2 [ t ] = FEVS [ t ] * (1.00 - SES_HCO2) * CES_HCO2
    # FEKS_H2CO2 [ t ] = FEVS [ t ] * (1.00 - SES_H2CO2) * CES_H2CO2
    FEKS_GLUC [ t ] = FEVS [ t ] * (1.00 - SES_GLUC) * CES_GLUC

    FIKM_NA [ t ] = FIVM [ t ] * (1.00 - SMI_NA) * CMI_NA
    FIKM_K [ t ] = FIVM [ t ] * (1.00 - SMI_K) * CMI_K
    FIKM_CL [ t ] = FIVM [ t ] * (1.00 - SMI_CL) * CMI_CL
    # FIKM_HCO3 [ t ] = FIVM [ t ] * (1.00 - SMI_HCO3) * CMI_HCO3
    # FIKM_H2CO3 [ t ] = FIVM [ t ] * (1.00 - SMI_H2CO3) * CMI_H2CO3
    # FIKM_CO2 [ t ] = FIVM [ t ] * (1.00 - SMI_CO2) * CMI_CO2
    # FIKM_HPO4 [ t ] = FIVM [ t ] * (1.00 - SMI_HPO4) * CMI_HPO4
    # FIKM_H2PO4 [ t ] = FIVM [ t ] * (1.00 - SMI_H2PO4) * CMI_H2PO4
    FIKM_UREA [ t ] = FIVM [ t ] * (1.00 - SMI_UREA) * CMI_UREA
    # FIKM_NH3 [ t ] = FIVM [ t ] * (1.00 - SMI_NH3) * CMI_NH3
    # FIKM_NH4 [ t ] = FIVM [ t ] * (1.00 - SMI_NH4) * CMI_NH4
    # FIKM_H [ t ] = FIVM [ t ] * (1.00 - SMI_H) * CMI_H
    # FIKM_HCO2 [ t ] = FIVM [ t ] * (1.00 - SMI_HCO2) * CMI_HCO2
    # FIKM_H2CO2 [ t ] = FIVM [ t ] * (1.00 - SMI_H2CO2) * CMI_H2CO2
    FIKM_GLUC [ t ] = FIVM [ t ] * (1.00 - SMI_GLUC) * CMI_GLUC

    FIKS_NA [ t ] = FIVS [ t ] * (1.00 - SIS_NA) * CIS_NA
    FIKS_K [ t ] = FIVS [ t ] * (1.00 - SIS_K) * CIS_K
    FIKS_CL [ t ] = FIVS [ t ] * (1.00 - SIS_CL) * CIS_CL
    # FIKS_HCO3 [ t ] = FIVS [ t ] * (1.00 - SIS_HCO3) * CIS_HCO3
    # FIKS_H2CO3 [ t ] = FIVS [ t ] * (1.00 - SIS_H2CO3) * CIS_H2CO3
    # FIKS_CO2 [ t ] = FIVS [ t ] * (1.00 - SIS_CO2) * CIS_CO2
    # FIKS_HPO4 [ t ] = FIVS [ t ] * (1.00 - SIS_HPO4) * CIS_HPO4
    # FIKS_H2PO4 [ t ] = FIVS [ t ] * (1.00 - SIS_H2PO4) * CIS_H2PO4
    FIKS_UREA [ t ] = FIVS [ t ] * (1.00 - SIS_UREA) * CIS_UREA
    # FIKS_NH3 [ t ] = FIVS [ t ] * (1.00 - SIS_NH3) * CIS_NH3
    # FIKS_NH4 [ t ] = FIVS [ t ] * (1.00 - SIS_NH4) * CIS_NH4
    # FIKS_H [ t ] = FIVS [ t ] * (1.00 - SIS_H) * CIS_H
    # FIKS_HCO2 [ t ] = FIVS [ t ] * (1.00 - SIS_HCO2) * CIS_HCO2
    # FIKS_H2CO2 [ t ] = FIVS [ t ] * (1.00 - SIS_H2CO2) * CIS_H2CO2
    FIKS_GLUC [ t ] = FIVS [ t ] * (1.00 - SIS_GLUC) * CIS_GLUC

    JK_NA [ t ] = JV [ t ] * (1.00 - SIS_NA) * CIE_NA
    JK_K [ t ] = JV [ t ] * (1.00 - SIS_K) * CIE_K
    JK_CL [ t ] = JV [ t ] * (1.00 - SIS_CL) * CIE_CL
    # JK_HCO3 [ t ] = JV [ t ] * (1.00 - SIS_HCO3) * CIE_HCO3
    # JK_H2CO3 [ t ] = JV [ t ] * (1.00 - SIS_H2CO3) * CIE_H2CO3
    # JK_CO2 [ t ] = JV [ t ] * (1.00 - SIS_CO2) * CIE_CO2
    # JK_HPO4 [ t ] = JV [ t ] * (1.00 - SIS_HPO4) * CIE_HPO4
    # JK_H2PO4 [ t ] = JV [ t ] * (1.00 - SIS_H2PO4) * CIE_H2PO4
    JK_UREA [ t ] = JV [ t ] * (1.00 - SIS_UREA) * CIE_UREA
    # JK_NH3 [ t ] = JV [ t ] * (1.00 - SIS_NH3) * CIE_NH3
    # JK_NH4 [ t ] = JV [ t ] * (1.00 - SIS_NH4) * CIE_NH4
    # JK_H [ t ] = JV [ t ] * (1.00 - SIS_H) * CIE_H
    # JK_HCO2 [ t ] = JV [ t ] * (1.00 - SIS_HCO2) * CIE_HCO2
    # JK_H2CO2 [ t ] = JV [ t ] * (1.00 - SIS_H2CO2) * CIE_H2CO2
    JK_GLUC [ t ] = JV [ t ] * (1.00 - SIS_GLUC) * CIE_GLUC

    # GOLDMAN  FLUXES
    FEKM_NA [ t ] = FEKM_NA [ t ] + GOLDMAN(HME_NA, AME, Z_NA, VM [ t ], VE [ t ], CM_NA,
                                            CE_NA [ t ], 1)
    FEKM_K [ t ] = FEKM_K [ t ] + GOLDMAN(HME_K, AME, Z_K, VM [ t ], VE [ t ], CM_K, CE_K [ t ], 1)
    FEKM_CL [ t ] = FEKM_CL [ t ] + GOLDMAN(HME_CL, AME, Z_CL, VM [ t ], VE [ t ], CM_CL, CE_CL [ t ], 1)
    # FEKM_HCO3 [ t ] = FEKM_HCO3 [ t ] + GOLDMAN(HME_HCO3, AME, Z_HCO3, VM [ t ], VE [ t ], CM_HCO3, CE_HCO3 [ t ],
    #                                             1)
    # FEKM_H2CO3 [ t ] = FEKM_H2CO3 [ t ] + GOLDMAN(HME_H2CO3, AME, Z_H2CO3, VM [ t ], VE [ t ], CM_H2CO3,
    #                                               CE_H2CO3 [ t ], 1)
    # FEKM_CO2 [ t ] = FEKM_CO2 [ t ] + GOLDMAN(HME_CO2, AME, Z_CO2, VM [ t ], VE [ t ], CM_CO2, CE_CO2 [ t ], 1)
    # FEKM_HPO4 [ t ] = FEKM_HPO4 [ t ] + GOLDMAN(HME_HPO4, AME, Z_HPO4, VM [ t ], VE [ t ], CM_HPO4,
    #                                             CE_HPO4 [ t ], 1)
    # FEKM_H2PO4 [ t ] = FEKM_H2PO4 [ t ] + GOLDMAN(HME_H2PO4, AME, Z_H2PO4, VM [ t ], VE [ t ], CM_H2PO4,
    #                                               CE_H2PO4 [ t ], 1)
    FEKM_UREA [ t ] = FEKM_UREA [ t ] + GOLDMAN(HME_UREA, AME, Z_UREA, VM [ t ], VE [ t ], CM_UREA, CE_UREA [ t ],
                                                1)
    # FEKM_NH3 [ t ] = FEKM_NH3 [ t ] + GOLDMAN(HME_NH3, AME, Z_NH3, VM [ t ], VE [ t ], CM_NH3, CE_NH3 [ t ], 1)
    # FEKM_NH4 [ t ] = FEKM_NH4 [ t ] + GOLDMAN(HME_NH4, AME, Z_NH4, VM [ t ], VE [ t ], CM_NH4, CE_NH4 [ t ], 1)
    # FEKM_H [ t ] = FEKM_H [ t ] + GOLDMAN(HME_H, AME, Z_H, VM [ t ], VE [ t ], CM_H, CE_H [ t ], 1)
    # FEKM_HCO2 [ t ] = FEKM_HCO2 [ t ] + GOLDMAN(HME_HCO2, AME, Z_HCO2, VM [ t ], VE [ t ], CM_HCO2, CE_HCO2 [ t ],
    #                                             1)
    # FEKM_H2CO2 [ t ] = FEKM_H2CO2 [ t ] + GOLDMAN(HME_H2CO2, AME, Z_H2CO2, VM [ t ], VE [ t ], CM_H2CO2,
    #                                               CE_H2CO2 [ t ], 1)
    FEKM_GLUC [ t ] = FEKM_GLUC [ t ] + GOLDMAN(HME_GLUC, AME, Z_GLUC, VM [ t ], VE [ t ], CM_GLUC, CE_GLUC [ t ], 1)

    FEKS_NA [ t ] = FEKS_NA [ t ] + GOLDMAN(HES_NA, AE [ t ], Z_NA, VE [ t ], VS, CE_NA [ t ],
                                            CS_NA, 1)
    FEKS_K [ t ] = FEKS_K [ t ] + GOLDMAN(HES_K, AE [ t ], Z_K, VE [ t ], VS, CE_K [ t ], CS_K, 1)
    FEKS_CL [ t ] = FEKS_CL [ t ] + GOLDMAN(HES_CL, AE [ t ], Z_CL, VE [ t ], VS, CE_CL [ t ], CS_CL, 1)
    # FEKS_HCO3 [ t ] = FEKS_HCO3 [ t ] + GOLDMAN(HES_HCO3, AE [ t ], Z_HCO3, VE [ t ], VS, CE_HCO3 [ t ],
    #                                             CS_HCO3, 1)
    # FEKS_H2CO3 [ t ] = FEKS_H2CO3 [ t ] + GOLDMAN(HES_H2CO3, AE [ t ], Z_H2CO3, VE [ t ], VS, CE_H2CO3 [ t ],
    #                                               CS_H2CO3, 1)
    # FEKS_CO2 [ t ] = FEKS_CO2 [ t ] + GOLDMAN(HES_CO2, AE [ t ], Z_CO2, VE [ t ], VS, CE_CO2 [ t ], CS_CO2, 1)
    # FEKS_HPO4 [ t ] = FEKS_HPO4 [ t ] + GOLDMAN(HES_HPO4, AE [ t ], Z_HPO4, VE [ t ], VS, CE_HPO4 [ t ],
    #                                             CS_HPO4, 1)
    # FEKS_H2PO4 [ t ] = FEKS_H2PO4 [ t ] + GOLDMAN(HES_H2PO4, AE [ t ], Z_H2PO4, VE [ t ], VS, CE_H2PO4 [ t ],
    #                                               CS_H2PO4, 1)
    FEKS_UREA [ t ] = FEKS_UREA [ t ] + GOLDMAN(HES_UREA, AE [ t ], Z_UREA, VE [ t ], VS, CE_UREA [ t ],
                                            CS_UREA, 1)
    # FEKS_NH3 [ t ] = FEKS_NH3 [ t ] + GOLDMAN(HES_NH3, AE [ t ], Z_NH3, VE [ t ], VS, CE_NH3 [ t ], CS_NH3, 1)
    # FEKS_NH4 [ t ] = FEKS_NH4 [ t ] + GOLDMAN(HES_NH4, AE [ t ], Z_NH4, VE [ t ], VS, CE_NH4 [ t ], CS_NH4, 1)
    # FEKS_H [ t ] = FEKS_H [ t ] + GOLDMAN(HES_H, AE [ t ], Z_H, VE [ t ], VS, CE_H [ t ], CS_H, 1)
    # FEKS_HCO2 [ t ] = FEKS_HCO2 [ t ] + GOLDMAN(HES_HCO2, AE [ t ], Z_HCO2, VE [ t ], VS, CE_HCO2 [ t ],CS_HCO2, 1)
    FEKS_GLUC [ t ] = FEKS_GLUC [ t ] + GOLDMAN(HES_GLUC, AE [ t ], Z_GLUC, VE [ t ], VS, CE_GLUC [ t ], CS_GLUC, 1)


    FIKM_NA [ t ] = FIKM_NA [ t ] + GOLDMAN(HMI_NA, AMI, Z_NA, VM [ t ], VI [ t ], CM_NA,
                                            CI_NA [ t ], 1)
    FIKM_K [ t ] = FIKM_K [ t ] + GOLDMAN(HMI_K, AMI, Z_K, VM [ t ], VI [ t ], CM_K, CI_K [ t ], 1)
    FIKM_CL [ t ] = FIKM_CL [ t ] + GOLDMAN(HMI_CL, AMI, Z_CL, VM [ t ], VI [ t ], CM_CL, CI_CL [ t ], 1)
    # FIKM_HCO3 [ t ] = FIKM_HCO3 [ t ] + GOLDMAN(HMI_HCO3, AMI, Z_HCO3, VM [ t ], VI [ t ], CM_HCO3, CI_HCO3 [ t ], 1)
    # FIKM_H2CO3 [ t ] = FIKM_H2CO3 [ t ] + GOLDMAN(HMI_H2CO3, AMI, Z_H2CO3, VM [ t ], VI [ t ], CM_H2CO3,
    #                                               CI_H2CO3 [ t ], 1)
    # FIKM_CO2 [ t ] = FIKM_CO2 [ t ] + GOLDMAN(HMI_CO2, AMI, Z_CO2, VM [ t ], VI [ t ], CM_CO2, CI_CO2 [ t ], 1)
    # FIKM_HPO4 [ t ] = FIKM_HPO4 [ t ] + GOLDMAN(HMI_HPO4, AMI, Z_HPO4, VM [ t ], VI [ t ], CM_HPO4,
    #                                             CI_HPO4 [ t ], 1)
    # FIKM_H2PO4 [ t ] = FIKM_H2PO4 [ t ] + GOLDMAN(HMI_H2PO4, AMI, Z_H2PO4, VM [ t ], VI [ t ], CM_H2PO4,
    #                                               CI_H2PO4 [ t ], 1)
    FIKM_UREA [ t ] = FIKM_UREA [ t ] + GOLDMAN(HMI_UREA, AMI, Z_UREA, VM [ t ], VI [ t ], CM_UREA, CI_UREA [ t ], 1)
    # FIKM_NH3 [ t ] = FIKM_NH3 [ t ] + GOLDMAN(HMI_NH3, AMI, Z_NH3, VM [ t ], VI [ t ], CM_NH3, CI_NH3 [ t ], 1)
    # FIKM_NH4 [ t ] = FIKM_NH4 [ t ] + GOLDMAN(HMI_NH4, AMI, Z_NH4, VM [ t ], VI [ t ], CM_NH4, CI_NH4 [ t ], 1)
    # FIKM_H [ t ] = FIKM_H [ t ] + GOLDMAN(HMI_H, AMI, Z_H, VM [ t ], VI [ t ], CM_H, CI_H [ t ], 1)
    # FIKM_HCO2 [ t ] = FIKM_HCO2 [ t ] + GOLDMAN(HMI_HCO2, AMI, Z_HCO2, VM [ t ], VI [ t ], CM_HCO2, CI_HCO2 [ t ], 1)
    # FIKM_H2CO2 [ t ] = FIKM_H2CO2 [ t ] + GOLDMAN(HMI_H2CO2, AMI, Z_H2CO2, VM [ t ], VI [ t ], CM_H2CO2,
    #                                               CI_H2CO2 [ t ], 1)
    FIKM_GLUC [ t ] = FIKM_GLUC [ t ] + GOLDMAN(HMI_GLUC, AMI, Z_GLUC, VM [ t ], VI [ t ], CM_GLUC, CI_GLUC [ t ], 1)


    JK_NA [ t ] = JK_NA [ t ] + GOLDMAN(HIS_NA, AIE, Z_NA, VI [ t ], VE [ t ], CI_NA [ t ],
                                        CE_NA [ t ], 1)
    JK_K [ t ] = JK_K [ t ] + GOLDMAN(HIS_K, AIE, Z_K, VI [ t ], VE [ t ], CI_K [ t ], CE_K [ t ], 1)
    JK_CL [ t ] = JK_CL [ t ] + GOLDMAN(HIS_CL, AIE, Z_CL, VI [ t ], VE [ t ], CI_CL [ t ], CE_CL [ t ], 1)
    # JK_HCO3 [ t ] = JK_HCO3 [ t ] + GOLDMAN(HIS_HCO3, AIE, Z_HCO3, VI [ t ], VE [ t ], CI_HCO3 [ t ], CE_HCO3 [ t ], 1)
    # JK_H2CO3 [ t ] = JK_H2CO3 [ t ] + GOLDMAN(HIS_H2CO3, AIE, Z_H2CO3, VI [ t ], VE [ t ], CI_H2CO3 [ t ],
    #                                           CE_H2CO3 [ t ], 1)
    # JK_CO2 [ t ] = JK_CO2 [ t ] + GOLDMAN(HIS_CO2, AIE, Z_CO2, VI [ t ], VE [ t ], CI_CO2 [ t ], CE_CO2 [ t ], 1)
    # JK_HPO4 [ t ] = JK_HPO4 [ t ] + GOLDMAN(HIS_HPO4, AIE, Z_HPO4, VI [ t ], VE [ t ], CI_HPO4 [ t ],
    #                                         CE_HPO4 [ t ], 1)
    # JK_H2PO4 [ t ] = JK_H2PO4 [ t ] + GOLDMAN(HIS_H2PO4, AIE, Z_H2PO4, VI [ t ], VE [ t ], CI_H2PO4 [ t ],
    #                                           CE_H2PO4 [ t ], 1)
    JK_UREA [ t ] = JK_UREA [ t ] + GOLDMAN(HIS_UREA, AIE, Z_UREA, VI [ t ], VE [ t ], CI_UREA [ t ], CE_UREA [ t ], 1)
    # JK_NH3 [ t ] = JK_NH3 [ t ] + GOLDMAN(HIS_NH3, AIE, Z_NH3, VI [ t ], VE [ t ], CI_NH3 [ t ], CE_NH3 [ t ], 1)
    # JK_NH4 [ t ] = JK_NH4 [ t ] + GOLDMAN(HIS_NH4, AIE, Z_NH4, VI [ t ], VE [ t ], CI_NH4 [ t ], CE_NH4 [ t ], 1)
    # JK_H [ t ] = JK_H [ t ] + GOLDMAN(HIS_H, AIE, Z_H, VI [ t ], VE [ t ], CI_H [ t ], CE_H [ t ], 1)
    # JK_HCO2 [ t ] = JK_HCO2 [ t ] + GOLDMAN(HIS_HCO2, AIE, Z_HCO2, VI [ t ], VE [ t ], CI_HCO2 [ t ], CE_HCO2 [ t ], 1)
    # JK_H2CO2 [ t ] = JK_H2CO2 [ t ] + GOLDMAN(HIS_H2CO2, AIE, Z_H2CO2, VI [ t ], VE [ t ], CI_H2CO2 [ t ],
    #                                           CE_H2CO2 [ t ], 1)
    JK_GLUC [ t ] = JK_GLUC [ t ] + GOLDMAN(HIS_GLUC, AIE, Z_GLUC, VI [ t ], VE [ t ], CI_GLUC [ t ], CE_GLUC [ t ], 1)

    FIKS_NA [ t ] = FIKS_NA [ t ] + GOLDMAN(HIS_NA, AIS, Z_NA, VI [ t ], VS, CI_NA [ t ],
                                            CS_NA, 1)
    FIKS_K [ t ] = FIKS_K [ t ] + GOLDMAN(HIS_K, AIS, Z_K, VI [ t ], VS, CI_K [ t ], CS_K, 1)
    FIKS_CL [ t ] = FIKS_CL [ t ] + GOLDMAN(HIS_CL, AIS, Z_CL, VI [ t ], VS, CI_CL [ t ], CS_CL, 1)
    # FIKS_HCO3 [ t ] = FIKS_HCO3 [ t ] + GOLDMAN(HIS_HCO3, AIS, Z_HCO3, VI [ t ], VS, CI_HCO3 [ t ], CS_HCO3, 1)
    # FIKS_H2CO3 [ t ] = FIKS_H2CO3 [ t ] + GOLDMAN(HIS_H2CO3, AIS, Z_H2CO3, VI [ t ], VS, CI_H2CO3 [ t ],
    #                                               CS_H2CO3, 1)
    # FIKS_CO2 [ t ] = FIKS_CO2 [ t ] + GOLDMAN(HIS_CO2, AIS, Z_CO2, VI [ t ], VS, CI_CO2 [ t ], CS_CO2, 1)
    # FIKS_HPO4 [ t ] = FIKS_HPO4 [ t ] + GOLDMAN(HIS_HPO4, AIS, Z_HPO4, VI [ t ], VS, CI_HPO4 [ t ],
    #                                             CS_HPO4, 1)
    # FIKS_H2PO4 [ t ] = FIKS_H2PO4 [ t ] + GOLDMAN(HIS_H2PO4, AIS, Z_H2PO4, VI [ t ], VS, CI_H2PO4 [ t ],
    #                                               CS_H2PO4, 1)
    FIKS_UREA [ t ] = FIKS_UREA [ t ] + GOLDMAN(HIS_UREA, AIS, Z_UREA, VI [ t ], VS, CI_UREA [ t ], CS_UREA, 1)
    # FIKS_NH3 [ t ] = FIKS_NH3 [ t ] + GOLDMAN(HIS_NH3, AIS, Z_NH3, VI [ t ], VS, CI_NH3 [ t ], CS_NH3, 1)
    # FIKS_NH4 [ t ] = FIKS_NH4 [ t ] + GOLDMAN(HIS_NH4, AIS, Z_NH4, VI [ t ], VS, CI_NH4 [ t ], CS_NH4, 1)
    # FIKS_H [ t ] = FIKS_H [ t ] + GOLDMAN(HIS_H, AIS, Z_H, VI [ t ], VS, CI_H [ t ], CS_H, 1)
    # FIKS_HCO2 [ t ] = FIKS_HCO2 [ t ] + GOLDMAN(HIS_HCO2, AIS, Z_HCO2, VI [ t ], VS, CI_HCO2 [ t ], CS_HCO2, 1)
    # FIKS_H2CO2 [ t ] = FIKS_H2CO2 [ t ] + GOLDMAN(HIS_H2CO2, AIS, Z_H2CO2, VI [ t ], VS, CI_H2CO2 [ t ],
    #                                               CS_H2CO2, 1)
    FIKS_GLUC [ t ] = FIKS_GLUC [ t ] + GOLDMAN(HIS_GLUC, AIS, Z_GLUC, VI [ t ], VS, CI_GLUC [ t ], CS_GLUC, 1)

    # Net Cotransporters

    SGLT = SGLT_MI(CM_NA, CI_NA [ t ], CM_GLUC, CI_GLUC [ t ], Z_NA, Z_GLUC, VM [ t ], VI [ t ], AMI, LMI_NAGLUC,
                   1)
    NA_MI_NAGLUC = SGLT [ 0 ]
    GLUC_MI_NAGLUC = SGLT [ 1 ]
    NAH2PO4 = 0#NAH2PO4_MI(CM_NA, CI_NA [ t ], CM_H2PO4, CI_H2PO4 [ t ], Z_NA, Z_H2PO4, VM [ t ], VI [ t ], AMI,  LMI_NAH2PO4, 1)
    # NA_MI_NAH2PO4 = NAH2PO4 [ 0 ]
    # H2PO4_MI_NAH2PO4 = NAH2PO4 [ 1 ]


    CLHCO3 =0

    KCL = KCL_IS(CI_K [ t ], CS_K, CI_CL [ t ], CS_CL, Z_K, Z_CL, VI [ t ], VS, AIS, LIS_KCL, 1)
    K_IS_KCL = KCL [ 0 ]
    CL_IS_KCL = KCL [ 1 ]


    # NA_CLHCO3 = NA_CLHCO3_IS(CI_NA [ t ], CS_NA, CI_CL [ t ], CS_CL, 0,0, Z_NA, Z_CL, 0,
    #                          VI [ t ], VS, AIS, LIS_NA_CLHCO3, 1)
    # NA_IS_NA_CLHCO3 = NA_CLHCO3 [ 0 ]
    # CL_IS_NA_CLHCO3 = NA_CLHCO3 [ 1 ]
    # HCO3_IS_NA_CLHCO3 = NA_CLHCO3 [ 2 ]
    # THE NAH EXCHANGER TRANSLATE CONCENTRATIONS TO THE NAH MODEL
    # MYNAH = NAH(CI_H [ t ], CI_NA [ t ], CI_NH4 [ t ], CM_H, CM_NA, CM_NH4, 1)
    # JNAH_NA = MYNAH [ 0 ]
    # JNAH_H = MYNAH [ 1 ]
    # JNAH_NH4 = MYNAH [ 2 ]
    # JNHE3_NA = NNHE3 * AMI * JNAH_NA
    # JNHE3_H = NNHE3 * AMI * JNAH_H
    # JNHE3_NH4 = NNHE3 * AMI * JNAH_NH4

    FIKM_NA [ t ] = FIKM_NA [ t ] + NA_MI_NAGLUC
    FIKM_CL [ t ] = FIKM_CL [ t ]
    # FIKM_HCO3 [ t ] = FIKM_HCO3 [ t ] + HCO3_MI_CLHCO3
    # FIKM_H2PO4 [ t ] = FIKM_H2PO4 [ t ] + H2PO4_MI_NAH2PO4
    # FIKM_HCO2 [ t ] = FIKM_HCO2 [ t ] + HCO2_MI_CLHCO2
    FIKM_GLUC [ t ] = FIKM_GLUC [ t ] + GLUC_MI_NAGLUC
    # FIKM_H [ t ] = FIKM_H [ t ] + JNHE3_H
    # FIKM_NH4 [ t ] = FIKM_NH4 [ t ] + JNHE3_NH4

    FIKS_NA [ t ] = FIKS_NA [ t ]
    FIKS_K [ t ] = FIKS_K [ t ] + K_IS_KCL
    FIKS_CL [ t ] = FIKS_CL [ t ] + CL_IS_KCL
    # FIKS_HCO3 [ t ] = FIKS_HCO3 [ t ] + HCO3_IS_NA_CLHCO3 + HCO3_IS_NAHCO3

    JK_NA [ t ] = JK_NA [ t ]
    JK_K [ t ] = JK_K [ t ] + K_IS_KCL
    JK_CL [ t ] = JK_CL [ t ] + CL_IS_KCL
   # JK_HCO3 [ t ] = JK_HCO3 [ t ] + HCO3_IS_NA_CLHCO3 + HCO3_IS_NAHCO3

    # SODIUM PUMPS
    NAK =NAK_ATP(CI_K [ t ], CS_K, CI_NA [ t ], CS_NA, CE_K [ t ], 0, 1)
    ATIS_NA = NAK [ 0 ]
    ATIS_K = NAK [ 1 ]
    ATIS_NH4 = NAK [ 2 ]
    #ATMI_H = AT_MI_H(CM_H, CI_H [ t ], VM [ t ], VI [ t ], Z_H, 1)

    JK_NA [ t ] = JK_NA [ t ] + AIE * ATIS_NA
    JK_K [ t ] = JK_K [ t ] + AIE * ATIS_K
    #JK_NH4 [ t ] = JK_NH4 [ t ] + AIE * ATIS_NH4
    FIKS_NA [ t ] = FIKS_NA [ t ] + AIS * ATIS_NA
    FIKS_K [ t ] = FIKS_K [ t ] + AIS * ATIS_K
    #FIKS_NH4 [ t ] = FIKS_NH4 [ t ] + AIS * ATIS_NH4

    # JNAK_NA= AIE*ATIS_NA+AIS*ATIS_NA
    # JNAK_K= AIE*ATIS_K+AIS*ATIS_K
    # JNAK_NH4= AIE*ATIS_NH4+AIS*ATIS_NH4

    # PROTON PUMPS
    #FIKM_H [ t ] = FIKM_H [ t ] + AMI * ATMI_H
    # ESTABLISH THE ERROR VECTORS, THE "PHI" ARRAY.

    # FIRST FOR THE INTERSPACE  ELECTRONEUTRALITY
    PHIE_EN = 0
    PHIE_EN = PHIE_EN + Z_NA * CE_NA [ t ] + Z_K * CE_K [ t ] + Z_CL * CE_CL [
        t ]+ Z_GLUC * CE_GLUC [t] + Z_UREA * CE_UREA [t]

    PHII_EN = IMP[t] * ZIMP
    PHII_EN = PHII_EN - CBUF [ t ] + Z_NA * CI_NA [ t ] + Z_K * CI_K [ t ] \
              + Z_CL * CI_CL [ t ] + Z_GLUC * CI_GLUC [t] + Z_UREA * CI_UREA [t]

    # MASS CONSERVATION IN THE TIME-DEPENDENT CASE
    PHIE_VLM = FEVS [ t ] - FEVM [ t ] + RTAU * (CHVL [ t ] - CHVL [ t - 1 ])
    QE_NA = FEKS_NA [ t ] - FEKM_NA [ t ] + RTAU * (CE_NA [ t ] * CHVL [ t ] - CE_NA [ t - 1 ] * CHVL [ t - 1 ])
    QE_K = FEKS_K [ t ] - FEKM_K [ t ] + RTAU * (CE_K [ t ] * CHVL [ t ] - CE_K [ t - 1 ] * CHVL [ t - 1 ])
    QE_CL = FEKS_CL [ t ] - FEKM_CL [ t ] + RTAU * (CE_CL [ t ] * CHVL [ t ] - CE_CL [ t - 1 ] * CHVL [ t - 1 ])
    QE_UREA = FEKS_UREA[ t ] - FEKM_UREA[ t ] + RTAU * (CE_UREA[ t ] * CHVL[ t ] - CE_UREA [ t - 1 ] * CHVL [ t - 1 ])
    QE_GLUC = FEKS_GLUC [ t ] - FEKM_GLUC [ t ] + RTAU * (CE_GLUC [ t ] * CHVL [ t ] - CE_GLUC [ t - 1 ] * CHVL [ t - 1 ])


    # MASS CONSERVATION IN THE TIME - DEPENDENT CASE
    PHII_VLM = FIVS [ t ] - FIVM [ t ] + JV [ t ] + RTAU * (CLVL [ t ] - CLVL [ t - 1 ])

    QI_NA = FIKS_NA [ t ] - FIKM_NA [ t ] + JK_NA [ t ] + RTAU * (
            CI_NA [ t ] * CLVL [ t ] - CI_NA [ t - 1 ] * CLVL [ t - 1 ])

    QI_K = FIKS_K [ t ] - FIKM_K [ t ] + JK_K [ t ] + RTAU * (
            CI_K [ t ] * CLVL [ t ] - CI_K [ t - 1 ] * CLVL [ t - 1 ])
    QI_CL = FIKS_CL [ t ] - FIKM_CL [ t ] + JK_CL [ t ] + RTAU * (
            CI_CL [ t ] * CLVL [ t ] - CI_CL [ t - 1 ] * CLVL [ t - 1 ])

    QI_UREA = FIKS_UREA[ t ] - FIKM_UREA[ t ] + JK_UREA[ t ] + RTAU * (
            CI_UREA[ t ] * CLVL[ t ] - CI_UREA [ T - 1 ] * CLVL [ T - 1 ])
    QI_GLUC = FIKS_GLUC [ t ] - FIKM_GLUC [ t ] + JK_GLUC [ t ] + RTAU * (
            CI_GLUC [ t ] * CLVL [ t ] - CI_GLUC [ t - 1 ] * CLVL [ t - 1 ])

        # THE PROTON FLUX MUST INCLUDE THE CELLULAR BUFFERS



    SOLS = 6
    kk = SOLS + 1
    Scale = 1e6
    PHIE_VLM = PHIScale(PHIE_VLM, Scale)
    QE_NA = PHIScale(QE_NA, Scale)
    QE_K = PHIScale(QE_K, Scale)
    QE_CL = PHIScale(QE_CL, Scale)
    QE_UREA = PHIScale(QE_UREA, Scale)
    QE_GLUC = PHIScale(QE_GLUC, Scale)

    PHII_VLM = PHIScale(PHII_VLM, Scale)
    QI_NA = PHIScale(QI_NA, Scale)
    QI_K = PHIScale(QI_K, Scale)
    QI_CL = PHIScale(QI_CL, Scale)
    QI_UREA = PHIScale(QI_UREA, Scale)
    QI_GLUC = PHIScale(QI_GLUC, Scale)
    #print('QI_NA', QI_NA)

    # NOW SET THE PHIS IN TERMS OF THE SOLUTE GENERATION RATES
    # FIRST THE NON-REACTIVE SPECIES
    PHI_ERROR = [ 0 for i in range(15)]
    PHI_ERROR [ -1 + 1 ] = PHIE_EN
    PHI_ERROR [ -1 + 2 ] = PHIE_VLM
    PHI_ERROR [ -1 + 3 ] = QE_NA
    PHI_ERROR [ -1 + 4 ] = QE_K
    PHI_ERROR [ -1 + 5 ] = QE_CL
    PHI_ERROR [ -1 + 6 ] = QE_UREA
    PHI_ERROR [ -1 + 7 ] = QE_GLUC
    PHI_ERROR [ -1 + 1 + kk ] = PHII_EN
    PHI_ERROR [ -1 + 2 + kk ] = PHII_VLM
    PHI_ERROR [ -1 + 3 + kk ] = QI_NA
    PHI_ERROR [ -1 + 4 + kk ] = QI_K
    PHI_ERROR [ -1 + 5 + kk ] = QI_CL
    PHI_ERROR [ -1 + 6 + kk ] = QI_UREA
    PHI_ERROR [ -1 + 7 + kk ] = QI_GLUC
    # CO2, FORMATE, PHOSPHATE, AND AMMONIA CONTENT:

    # CELL BUFFER CONTENT AND PH EQUILIBRIUM
    CURE = F * (
            Z_NA * FEKM_NA [ t ] + Z_K * FEKM_K [ t ] + Z_CL * FEKM_CL [ t ]  + Z_UREA * FEKM_UREA [ t ] + Z_GLUC * FEKM_GLUC [ t ])

    PHIE_CUR = CURE

    CURI = F * (
            Z_NA * FIKM_NA [ t ] + Z_K * FIKM_K [ t ] + Z_CL * FIKM_CL [ t ]  + Z_UREA * FIKM_UREA [ t ] + Z_GLUC * FIKM_GLUC [ t ])

    PHI_ERROR [ -1 +8 + kk ] = PHIE_CUR + CURI
    # print('CURI', CURI)
    # print('CURE ', CURE )
    return PHI_ERROR

t0 = 0
tf =800
T = 4000
# DX = (xf - x0)
# RTAU=0

DT = (tf - t0) / (T - 1)
RTAU = DT


def Matrix(init, h):
    return [ init for x in range(h) ]

# def calculate(X, T):
CME_NA_P = Matrix(0.2000e-01, T)
CME_NA = Matrix(0.2000e-01, T)
CME_K_P = Matrix(0.2000e-01, T)
CME_K = Matrix(0.2000e-01, T)
CME_CL_P = Matrix(0.2000e-01, T)
CME_CL = Matrix(0.2000e-01, T)
CME_HCO3_P = Matrix(0.2000e-01, T)
CME_HCO3 = Matrix(0.2000e-01, T)
CME_H2CO3_P = Matrix(0.2000e-01, T)
CME_H2CO3 = Matrix(0.2000e-01, T)
CME_CO2_P = Matrix(0.0002000e-01, T)
CME_CO2 = Matrix(0.2000e-01, T)
CME_HPO4_P = Matrix(0.2000e-01, T)
CME_HPO4 = Matrix(0.2000e-01, T)
CME_H2PO4_P = Matrix(0.2000e-01, T)
CME_H2PO4 = Matrix(0.2000e-01, T)
CME_UREA_P = Matrix(0.2000e-01, T)
CME_UREA = Matrix(0.2000e-01, T)
CME_NH3_P = Matrix(0.2000e-01, T)
CME_NH3 = Matrix(0.2000e-01, T)
CME_NH4_P = Matrix(0.2000e-01, T)
CME_NH4 = Matrix(0.2000e-01, T)
CME_H_P = Matrix(0.2000e-01, T)
CME_H = Matrix(0.2000e-01, T)
CME_HCO2_P = Matrix(0.2000e-01, T)
CME_HCO2 = Matrix(0.2000e-01, T)
CME_H2CO2_P = Matrix(0.2000e-01, T)
CME_H2CO2 = Matrix(0.2000e-01, T)
CME_GLUC_P = Matrix(0.2000e-01, T)
CME_GLUC = Matrix(0.2000e-01, T)

CES_NA_P = Matrix(0.2000e-01, T)
CES_NA = Matrix(0.2000e-01, T)
CES_K_P = Matrix(0.2000e-01, T)
CES_K = Matrix(0.2000e-01, T)
CES_CL_P = Matrix(0.2000e-01, T)
CES_CL = Matrix(0.2000e-01, T)
CES_HCO3_P = Matrix(0.2000e-01, T)
CES_HCO3 = Matrix(0.2000e-01, T)
CES_H2CO3_P = Matrix(0.2000e-01, T)
CES_H2CO3 = Matrix(0.2000e-01, T)
CES_CO2_P = Matrix(0.2000e-01, T)
CES_CO2 = Matrix(0.2000e-01, T)
CES_HPO4_P = Matrix(0.2000e-01, T)
CES_HPO4 = Matrix(0.2000e-01, T)
CES_H2PO4_P = Matrix(0.2000e-01, T)
CES_H2PO4 = Matrix(0.2000e-01, T)
CES_UREA_P = Matrix(0.2000e-01, T)
CES_UREA = Matrix(0.2000e-01, T)
CES_NH3_P = Matrix(0.2000e-01, T)
CES_NH3 = Matrix(0.2000e-01, T)
CES_NH4_P = Matrix(0.2000e-01, T)
CES_NH4 = Matrix(0.2000e-01, T)
CES_H_P = Matrix(0.2000e-01, T)
CES_H = Matrix(0.2000e-01, T)
CES_HCO2_P = Matrix(0.2000e-01, T)
CES_HCO2 = Matrix(0.2000e-01, T)
CES_H2CO2_P = Matrix(0.2000e-01, T)
CES_H2CO2 = Matrix(0.2000e-01, T)
CES_GLUC_P = Matrix(0.2000e-01, T)
CES_GLUC = Matrix(0.2000e-01, T)
CMI_NA_P = Matrix(0.2000e-01, T)
CMI_NA = Matrix(0.2000e-01, T)
CMI_K_P = Matrix(0.2000e-01, T)
CMI_K = Matrix(0.2000e-01, T)
CMI_CL_P = Matrix(0.2000e-01, T)
CMI_CL = Matrix(0.2000e-01, T)
CMI_HCO3_P = Matrix(0.2000e-01, T)
CMI_HCO3 = Matrix(0.2000e-01, T)
CMI_H2CO3_P = Matrix(0.2000e-01, T)
CMI_H2CO3 = Matrix(0.2000e-01, T)
CMI_CO2_P = Matrix(0.2000e-01, T)
CMI_CO2 = Matrix(0.2000e-01, T)
CMI_HPO4_P = Matrix(0.2000e-01, T)
CMI_HPO4 = Matrix(0.2000e-01, T)
CMI_H2PO4_P = Matrix(0.2000e-01, T)
CMI_H2PO4 = Matrix(0.2000e-01, T)
CMI_UREA_P = Matrix(0.2000e-01, T)
CMI_UREA = Matrix(0.2000e-01, T)
CMI_NH3_P = Matrix(0.2000e-01, T)
CMI_NH3 = Matrix(0.2000e-01, T)
CMI_NH4_P = Matrix(0.2000e-01, T)
CMI_NH4 = Matrix(0.2000e-01, T)
CMI_H_P = Matrix(0.2000e-01, T)
CMI_H = Matrix(0.2000e-01, T)
CMI_HCO2_P = Matrix(0.2000e-01, T)
CMI_HCO2 = Matrix(0.2000e-01, T)
CMI_H2CO2_P = Matrix(0.2000e-01, T)
CMI_H2CO2 = Matrix(0.2000e-01, T)
CMI_GLUC_P = Matrix(0.2000e-01, T)
CMI_GLUC = Matrix(0.2000e-01, T)
CIS_NA_P = Matrix(0.2000e-01, T)
CIS_NA = Matrix(0.2000e-01, T)
CIS_K_P = Matrix(0.2000e-01, T)
CIS_K = Matrix(0.2000e-01, T)
CIS_CL_P = Matrix(0.2000e-01, T)
CIS_CL = Matrix(0.2000e-01, T)
CIS_HCO3_P = Matrix(0.2000e-01, T)
CIS_HCO3 = Matrix(0.2000e-01, T)
CIS_H2CO3_P = Matrix(0.2000e-01, T)
CIS_H2CO3 = Matrix(0.2000e-01, T)
CIS_CO2_P = Matrix(0.2000e-01, T)
CIS_CO2 = Matrix(0.2000e-01, T)
CIS_HPO4_P = Matrix(0.2000e-01, T)
CIS_HPO4 = Matrix(0.2000e-01, T)
CIS_H2PO4_P = Matrix(0.2000e-01, T)
CIS_H2PO4 = Matrix(0.2000e-01, T)
CIS_UREA_P = Matrix(0.2000e-01, T)
CIS_UREA = Matrix(0.2000e-01, T)
CIS_NH3_P = Matrix(0.2000e-01, T)
CIS_NH3 = Matrix(0.2000e-01, T)
CIS_NH4_P = Matrix(0.2000e-01, T)
CIS_NH4 = Matrix(0.2000e-01, T)
CIS_H_P = Matrix(0.2000e-01, T)
CIS_H = Matrix(0.2000e-01, T)
CIS_HCO2_P = Matrix(0.2000e-01, T)
CIS_HCO2 = Matrix(0.2000e-01, T)
CIS_H2CO2_P = Matrix(0.2000e-01, T)
CIS_H2CO2 = Matrix(0.2000e-01, T)
CIS_GLUC_P = Matrix(0.2000e-01, T)
CIS_GLUC = Matrix(0.2000e-01, T)
CIE_NA_P = Matrix(0.2000e-01, T)
CIE_NA = Matrix(0.2000e-01, T)
CIE_K_P = Matrix(0.2000e-01, T)
CIE_K = Matrix(0.2000e-01, T)
CIE_CL_P = Matrix(0.2000e-01, T)
CIE_CL = Matrix(0.2000e-01, T)
CIE_HCO3_P = Matrix(0.2000e-01, T)
CIE_HCO3 = Matrix(0.2000e-01, T)
CIE_H2CO3_P = Matrix(0.2000e-01, T)
CIE_H2CO3 = Matrix(0.2000e-01, T)
CIE_CO2_P = Matrix(0.2000e-01, T)
CIE_CO2 = Matrix(0.2000e-01, T)
CIE_HPO4_P = Matrix(0.2000e-01, T)
CIE_HPO4 = Matrix(0.2000e-01, T)
CIE_H2PO4_P = Matrix(0.2000e-01, T)
CIE_H2PO4 = Matrix(0.2000e-01, T)
CIE_UREA_P = Matrix(0.2000e-01, T)
CIE_UREA = Matrix(0.2000e-01, T)
CIE_NH3_P = Matrix(0.2000e-01, T)
CIE_NH3 = Matrix(0.2000e-01, T)
CIE_NH4_P = Matrix(0.2000e-01, T)
CIE_NH4 = Matrix(0.2000e-01, T)
CIE_H_P = Matrix(0.2000e-01, T)
CIE_H = Matrix(0.2000e-01, T)
CIE_HCO2_P = Matrix(0.2000e-01, T)
CIE_HCO2 = Matrix(0.2000e-01, T)
CIE_H2CO2_P = Matrix(0.2000e-01, T)
CIE_H2CO2 = Matrix(0.2000e-01, T)
CIE_GLUC_P = Matrix(0.2000e-01, T)
CIE_GLUC = Matrix(0.2000e-01, T)
AE = Matrix(0.2000, T)
VE = Matrix(-0.0100, T)
PE = Matrix(9.96, T)
CE_NA = Matrix(0.14400, T)
CE_K = Matrix(0.005, T)
CE_CL = Matrix(0.11856, T)
CE_HCO3 = Matrix(0.025, T)
CE_H2CO3 = Matrix(0.000004412, T)
CE_CO2 = Matrix(0.00151, T)
CE_HPO4 = Matrix(0.0021, T)
CE_H2PO4 = Matrix(0.00056, T)
CE_UREA = Matrix(0.0051, T)
CE_NH3 = Matrix(0.0000029, T)
CE_NH4 = Matrix(0.0001970631, T)
CE_HCO2 = Matrix(0.001, T)
CE_H2CO2 = Matrix(0.000000273, T)
CE_GLUC = Matrix(0.0051, T)
VI = Matrix(-55.2700, T)
PI = 0
CI_NA = Matrix(0.1440, T)
CI_K = Matrix(0.005, T)
CI_CL = Matrix(0.118596593, T)
CI_HCO3 = Matrix(0.025, T)
CI_H2CO3 = Matrix(0.000004412, T)
CI_CO2 = Matrix(0.0015, T)
CI_HPO4 = Matrix(0.002, T)
CI_H2PO4 = Matrix(0.0006, T)
CI_UREA = Matrix(0.005, T)
CI_NH3 = Matrix(0.000002937, T)
CI_NH4 = Matrix(0.0002, T)
CI_HCO2 = Matrix(0.001, T)
CI_H2CO2 = Matrix(0.000000273, T)
CI_GLUC = Matrix(0.005, T)
VM = Matrix(0, T)
PM = 15
CM_NA = 0.1
CM_K = 0.001
CM_CL = 0.1
CM_HCO3 = 0.02
CM_H2CO3 = 0.001
CM_CO2 = 0.002
CM_HPO4 = 0.004
CM_H2PO4 = 0.0008
CM_UREA = 0.0051
CM_NH3 = 0.000005
CM_NH4 = 0.0003
CM_HCO2 = 0.002
CM_H2CO2 = 0.000001
CM_GLUC = 0.004
CE_H = Matrix(4.59e-11, T)
CI_H = Matrix(4.69e-11, T)
CM_H = 4.95e-11
CS_H = 4.95e-11
PS = 9
VS = 0.0
CHVL = Matrix(0.7000e-04, T)
CLVL = Matrix(0.1000e-02, T)
L = Matrix(0.1000e-01, T)
IMP = Matrix(0.6000e-01, T)
HCBUF = Matrix(0.04001207867399, T)
CBUF = Matrix( 0.026959905, T)

RM = Matrix(0.1060e-02, T)
AM = Matrix(0, T)
FEVS = Matrix(0.01, T)
FIVM = Matrix(0.1, T)
FIVS = Matrix(0, T)
JV = Matrix(0, T)
LCHM = Matrix(0.3, T)
IMPS = 0.002
IMPM = 0
FEVM = Matrix(0.9, T)

CS_NA = 0.144
CS_K = 0.005
CS_CL = 0.1186
CS_HCO3 = 0.025
CS_H2CO3 = 0.000004411
CS_CO2 = 0.0015
CS_HPO4 = 0.002000477
CS_H2PO4 = 0.000599523
CS_UREA = 0.005
CS_NH3 = 0.000002937
CS_NH4 = 0.000197063
CS_HCO2 = 0.001
CS_H2CO2 = 0.000000273
CS_GLUC = 0.005

FEKM_NA = Matrix(0.01, T)
FEKM_K = Matrix(0.01, T)
FEKM_CL = Matrix(0.01, T)
FEKM_HCO3 = Matrix(0.01, T)
FEKM_H2CO3 = Matrix(0.01, T)
FEKM_CO2 = Matrix(0.01, T)
FEKM_HPO4 = Matrix(0.01, T)
FEKM_H2PO4 = Matrix(0.01, T)
FEKM_UREA = Matrix(0.01, T)
FEKM_NH3 = Matrix(0.01, T)
FEKM_NH4 = Matrix(0.01, T)
FEKM_H = Matrix(0.01, T)
FEKM_HCO2 = Matrix(0.01, T)
FEKM_H2CO2 = Matrix(0.01, T)
FEKM_GLUC = Matrix(0.01, T)

FEKS_NA = Matrix(0.01, T)
FEKS_K = Matrix(0.01, T)
FEKS_CL = Matrix(0.01, T)
FEKS_HCO3 = Matrix(0.01, T)
FEKS_H2CO3 = Matrix(0.01, T)
FEKS_CO2 = Matrix(0.01, T)
FEKS_HPO4 = Matrix(0.01, T)
FEKS_H2PO4 = Matrix(0.01, T)
FEKS_UREA = Matrix(0.01, T)
FEKS_NH3 = Matrix(0.01, T)
FEKS_NH4 = Matrix(0.01, T)
FEKS_H = Matrix(0.01, T)
FEKS_HCO2 = Matrix(0.01, T)
FEKS_H2CO2 = Matrix(0.01, T)
FEKS_GLUC = Matrix(0.01, T)

# FI VM[t]_NA = Matrix(0.01,  T)
# FI VM[t]_K = Matrix(0.01,  T)
# FI VM[t]_CL = Matrix(0.01,  T)
# FI VM[t]_HCO3 = Matrix(0.01,  T)
# FI VM[t]_H2CO3 = Matrix(0.01,  T)
# FI VM[t]_CO2 = Matrix(0.01,  T)
# FI VM[t]_HPO4 = Matrix(0.01,  T)
# FI VM[t]_H2PO4 = Matrix(0.01,  T)
# FI VM[t]_UREA = Matrix(0.01,  T)
# FI VM[t]_NH3 = Matrix(0.01,  T)
# FI VM[t]_NH4 = Matrix(0.01,  T)
# FI VM[t]_H = Matrix(0.01,  T)
# FI VM[t]_HCO2 = Matrix(0.01,  T)
# FI VM[t]_H2CO2 = Matrix(0.01,  T)
# FI VM[t]_GLUC = Matrix(0.01,  T)

FIKM_NA = Matrix(0.01, T)
FIKM_K = Matrix(0.01, T)
FIKM_CL = Matrix(0.01, T)
FIKM_HCO3 = Matrix(0.01, T)
FIKM_H2CO3 = Matrix(0.01, T)
FIKM_CO2 = Matrix(0.01, T)
FIKM_HPO4 = Matrix(0.01, T)
FIKM_H2PO4 = Matrix(0.01, T)
FIKM_UREA = Matrix(0.01, T)
FIKM_NH3 = Matrix(0.01, T)
FIKM_NH4 = Matrix(0.01, T)
FIKM_H = Matrix(0.01, T)
FIKM_HCO2 = Matrix(0.01, T)
FIKM_H2CO2 = Matrix(0.01, T)
FIKM_GLUC = Matrix(0.01, T)

FIKS_NA = Matrix(0.01, T)
FIKS_K = Matrix(0.01, T)
FIKS_CL = Matrix(0.01, T)
FIKS_HCO3 = Matrix(0.01, T)
FIKS_H2CO3 = Matrix(0.01, T)
FIKS_CO2 = Matrix(0.01, T)
FIKS_HPO4 = Matrix(0.01, T)
FIKS_H2PO4 = Matrix(0.01, T)
FIKS_UREA = Matrix(0.01, T)
FIKS_NH3 = Matrix(0.01, T)
FIKS_NH4 = Matrix(0.01, T)
FIKS_H = Matrix(0.01, T)
FIKS_HCO2 = Matrix(0.01, T)
FIKS_H2CO2 = Matrix(0.01, T)
FIKS_GLUC = Matrix(0.01, T)

JK_NA = Matrix(0.01, T)
JK_K = Matrix(0.01, T)
JK_CL = Matrix(0.01, T)
JK_HCO3 = Matrix(0.01, T)
JK_H2CO3 = Matrix(0.01, T)
JK_CO2 = Matrix(0.01, T)
JK_HPO4 = Matrix(0.01, T)
JK_H2PO4 = Matrix(0.01, T)
JK_UREA = Matrix(0.01, T)
JK_NH3 = Matrix(0.01, T)
JK_NH4 = Matrix(0.01, T)
JK_H = Matrix(0.01, T)
JK_HCO2 = Matrix(0.01, T)
JK_H2CO2 = Matrix(0.01, T)
JK_GLUC = Matrix(0.01, T)

JHP_H = Matrix(1, T)
FVM = Matrix(0.1200e-01, T)
FKM_NA = Matrix(1, T)
FKM_K = Matrix(1, T)
FKM_CL = Matrix(1, T)
FKM_HCO3 = Matrix(1, T)
FKM_H2CO3 = Matrix(0.01, T)
FKM_CO2 = Matrix(0.01, T)
FKM_HPO4 = Matrix(0.01, T)
FKM_H2PO4 = Matrix(0.01, T)
FKM_UREA = Matrix(0.01, T)
FKM_NH3 = Matrix(0.01, T)
FKM_NH4 = Matrix(0.01, T)
FKM_H = Matrix(0.01, T)
FKM_HCO2 = Matrix(0.01, T)
FKM_H2CO2 = Matrix(0.01, T)
FKM_GLUC = Matrix(0.01, T)
FKM_IMPM = Matrix(0.01, T)
PHIE_EN = Matrix(0.0, T)
PHIE = Matrix(0.0, T)
PHIE_NA = Matrix(0.1, T)
PHIE_K = Matrix(0.1, T)
PHIE_CL = Matrix(0, T)
PHIE_HCO3 = Matrix(0.1, T)
PHIE_H2CO3 = Matrix(0.1, T)
PHIE_CO2 = Matrix(0, T)
PHIE_HPO4 = Matrix(0, T)
PHIE_H2PO4 = Matrix(0, T)
PHIE_UREA = Matrix(0, T)
PHIE_NH3 = Matrix(0.1, T)
PHIE_NH4 = Matrix(0.1, T)
PHIE_H = Matrix(0, T)
PHIE_HCO2 = Matrix(0, T)
PHIE_H2CO2 = Matrix(0, T)
PHIE_GLUC = Matrix(0, T)

PHII_EN = Matrix(0, T)

PHII_VLM = Matrix(0, T)
PHII_NA = Matrix(0, T)
PHII_K = Matrix(0, T)
PHII_CL = Matrix(0, T)
PHII_HCO3 = Matrix(0, T)
PHII_H2CO3 = Matrix(0, T)
PHII_CO2 = Matrix(0, T)
PHII_HPO4 = Matrix(0, T)
PHII_H2PO4 = Matrix(0, T)
PHII_UREA = Matrix(0, T)
PHII_NH3 = Matrix(0, T)
PHII_NH4 = Matrix(0, T)
PHII_H = Matrix(0, T)
PHII_HCO2 = Matrix(0, T)
PHII_H2CO2 = Matrix(0, T)
PHII_GLUC = Matrix(0, T)

PHIM = Matrix(0, T)
PHIM_NA = Matrix(0, T)
PHIM_K = Matrix(0, T)
PHIM_CL = Matrix(0, T)
PHIM_HCO3 = Matrix(0, T)
PHIM_H2CO3 = Matrix(0, T)
PHIM_CO2 = Matrix(0, T)
PHIM_HPO4 = Matrix(0, T)
PHIM_H2PO4 = Matrix(0, T)
PHIM_UREA = Matrix(0, T)
PHIM_NH3 = Matrix(0, T)
PHIM_NH4 = Matrix(0, T)
PHIM_H = Matrix(0, T)
PHIM_HCO2 = Matrix(0, T)
PHIM_H2CO2 = Matrix(0, T)
PHIM_GLUC = Matrix(0, T)
LCHE = Matrix(0, T)
LCHI = Matrix(0, T)
PHI_CUR = Matrix(0, T)
CURE = Matrix(0, T)
CURI = Matrix(0, T)
PHIM_IMPM = Matrix(0, T)
PHIM_PF = Matrix(0, T)
PHII_CUR = Matrix(0, T)
PHIE_CUR = Matrix(0, T)

myList=[0.1 for i in range(T)]




t=0
GUESS = [ VE [t], PE [t], CE_NA [t], CE_K [t],
              CE_CL [t],  CE_UREA [t],CE_GLUC [t],
              VI [t], IMP[t], CI_NA [t], CI_K [t],
              CI_CL [t],  CI_UREA [t], CI_GLUC [t],
              VM [t] ]
print(type(GUESS),GUESS)


CMNA = [0.1 ]

while True:
    t += 1

    if t == T:
        break
    else:

        RESULT = fsolve(EQS, np.array(GUESS))
        GUESS = [ VE [ t -1], PE [ t -1], CE_NA [ t -1], CE_K [ t-1 ],
              CE_CL [ t-1 ],  CE_UREA [ t-1 ], CE_GLUC [ t-1 ],
              VI[ t-1 ], IMP[t-1], CI_NA [ t-1 ], CI_K [ t-1 ],
              CI_CL [ t-1 ],  CI_UREA [ t-1 ], CI_GLUC [ t-1 ],
              VM [ t -1] ]


t0 = 0
tf = 800
T = 4000
t = np.linspace(t0, tf, T)
import numpy as np
CE_GLUC=np.asarray(CE_GLUC)
t_size = len(t)
GLUC_size = len(CE_GLUC)
# replicate = tf
# with open("output_data_VM"+ str(replicate) + ".csv", "w") as out_file:
#     for i in range(len(VM)):
#         OUT_VM =""
#         OUT_VM += str(VM[i])
#         OUT_VM = "\n"
#         out_file.write(OUT_VM)
#


import matplotlib.pyplot as plt
# Add title and axis names
import matplotlib.ticker as mticker
# plt.figure(1)
# plt.plot(t, CI_CL,  linewidth=3.0)
# # plt.ylabel('GLUC')
# # plt.xlabel('t')
# #title.set_text('My title')
#
# # %exit()
# plt.figure(2)
# plt.plot( CI_NA, linewidth=3.0)
# # plt.ylabel('NA')
# # plt.xlabel('t')
# #title.set_text('My title')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fig2 = plt.figure(constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2)
ax1 = fig2.add_subplot(spec2[0,0])
ax1.plot(t[0:-2], myList[0:-2], 'r-')
ax1.set_ylabel('CM_Na (mM)', color='blue')
ax2 = fig2.add_subplot(spec2[1,0])
ax2.plot(t, CI_NA, 'b-')
ax2.set_ylabel(' CI_NA ', color='blue')
ax3 = fig2.add_subplot(spec2[2,0])
ax3.plot(t, CI_K, 'b-')
ax3.set_ylabel(' CI_K ', color='blue')
ax4 = fig2.add_subplot(spec2[3,0])
ax4.plot(t, CI_CL, 'b-')
#f2_ax3 = fig2.add_subplot(spec2[01, 0])
ax4.set_ylabel('CI_Cl ', color='blue')
ax4.set_xlabel('Time (s) ', color='blue')

ax5 = fig2.add_subplot(spec2[4,0])
ax5.plot(t, CI_UREA, 'b-')
#f2_ax3 = fig2.add_subplot(spec2[01, 0])
ax5.set_ylabel('CI_UREA ', color='blue')
ax5.set_xlabel('Time (s) ', color='blue')
plt.show()
for ax in fig2.get_axes():

    ax.label_outer()# #


#
# fig = plt.figure(13)
# ax1 = fig.add_subplot(1, 1, 1)
# ax2 = ax1.twinx()
#
# ax1.plot(t, CI_NA, 'b-',label='CI_NA')
# ax2.plot(t, CI_K, 'r-',label='CI_K')
# ax1.set_ylabel('Na', color='blue')
# ax2.set_ylabel('K', color='red' )
# ax1.set_xlabel('Time (s)')
# ax1.set_xlim(0,tf)
# ax1.title.set_text('Na/K-ATPase')
#
# ax1.legend(loc=" ", bbox_to_anchor=(1,0.95), bbox_transform=ax1.transAxes)
# ax2.legend(loc=" ", bbox_to_anchor=(1,0.87), bbox_transform=ax2.transAxes)
# fig = plt.figure(14)
# ax1 = fig.add_subplot(1, 1, 1)
# ax2 = ax1.twinx()
# ax1.plot(t, CI_NA, 'b-',label='CI_NA')
# ax2.plot(t, CI_GLUC, 'r-',label='CI_GLUC')
# ax1.set_ylabel('Na', color='blue')
# ax2.set_ylabel('Glucose', color='red' )
# ax1.set_xlabel('Time (s)')
# ax1.set_xlim(0,tf)
# ax1.title.set_text('Na-Glucose Cotransporter')
# ax1.legend(loc=" ", bbox_to_anchor=(1,0.95), bbox_transform=ax1.transAxes)
# ax2.legend(loc=" ", bbox_to_anchor=(1,0.87), bbox_transform=ax2.transAxes)
# fig = plt.figure(18)
# ax1 = fig.add_subplot(1, 1, 1)
# ax2 = ax1.twinx()
#
# ax1.plot(t, CI_K, 'b-',label='CI_K')
# ax2.plot(t, CI_CL, 'r-',label='CI_Cl' )
# ax1.set_ylabel('K', color='blue')
# ax2.set_ylabel('Cl', color='red')
# ax1.set_xlabel('Time (s)')
# ax1.set_xlim(0,tf)
# ax1.title.set_text('K/Cl Cotransporter')
# ax1.legend(loc=" ", bbox_to_anchor=(1,0.95), bbox_transform=ax1.transAxes)
# ax2.legend(loc=" ", bbox_to_anchor=(1,0.87), bbox_transform=ax2.transAxes)
# plt.show()
#
#
# # fig = plt.figure(15)
# # ax1 = fig.add_subplot(1, 1, 1)
# # ax2 = ax1.twinx()
# #
# # ax1.plot(t, CE_NA, 'b-')
# # ax2.plot(t, CE_K, 'r-')
# # ax1.set_ylabel('Na', color='blue')
# # ax2.set_ylabel('K', color='red' )
# # ax1.set_xlabel('Time (s)')
# #
# # fig = plt.figure(16)
# # ax1 = fig.add_subplot(1, 1, 1)
# # ax2 = ax1.twinx()
# #
# # ax1.plot(t, CE_NA, 'b-')
# # ax2.plot(t, CE_GLUC, 'r-')
# # ax1.set_ylabel('Na', color='blue')
# # ax2.set_ylabel('glucose', color='red' )
# # ax1.set_xlabel('Time (s)')
# #
# # fig = plt.figure(17)
# # ax1 = fig.add_subplot(1, 1, 1)
# # ax2 = ax1.twinx()
# #
# # ax1.plot(t, CE_K, 'b-')
# # ax2.plot(t, CE_CL, 'r-')
# # ax1.set_ylabel('K', color='blue')
# # ax2.set_ylabel('Cl', color='red' )
# # ax1.set_xlabel('Time (s)')
# #