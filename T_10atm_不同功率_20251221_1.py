import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as kB, h, c, e
import math
from collections import namedtuple
from scipy.ndimage import gaussian_filter1d
import os
import numpy as np

# ---- numpy 版本兼容：trapz <-> trapezoid ----
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
if not hasattr(np, "trapezoid") and hasattr(np, "trapz"):
    np.trapezoid = np.trapz

plt.rcParams['font.sans-serif'] = ['SimSun']     # 支持中文的宋体
plt.rcParams['axes.unicode_minus'] = False       # 解决负号显示问题

# ---------------- 全局物理参数 ----------------
P_BAR  = 10.0          # bar
T_G    = 9000.0        # K（重粒子）
T_RAD  = 9000.0        # K（辐射温度，用于 B_lambda）
LAM_DISPLAY = (200, 1000)  # nm
RES_FWHM_NM = 1.8           # nm 仪器函数

# ✅要输出的激光功率列表（W）
P_LASER_LIST = [200, 450, 900, 1800, 2700, 3600, 4500, 5400, 6300]

# 三温区：稍微有中心高温区
ZONES = [
    dict(Te_fac=0.8, ne_fac=0.4, L_m=0.0015),
    dict(Te_fac=1.0, ne_fac=1.0, L_m=0.0030),
    dict(Te_fac=0.9, ne_fac=0.7, L_m=0.0015),
]

# Xe 电离能与主激发能量
E_ION = np.array([12.13, 21.21, 32.12, 45.14])
E_EXC = 8.4

# 光谱网格
wav_nm = np.linspace(200, 1000, 4001)
lam_m  = wav_nm * 1e-9

# 物理常数
pi = math.pi
m_e = 9.10938356e-31
epsilon_0 = 8.8541878128e-12

# ---------------- 标度系数（目前比较合适的一组） ----------------
FF_SCALE         = 2e-10   # ff+fb 连续谱整体缩放（极小 → Xe2* 占主导）
EXC_SCALE        = 5e-38   # Xe2* 连续谱强度缩放（乘 ne_m3*n0_m3）
KAPPA_CONT_SCALE = 0.01    # 连续谱吸收系数缩放
LINE_SCALE       = 320     # 线谱强度缩放：只做小尖峰
ALPHA_LINE       = 0.01    # 线自吸收强度系数（0~1，小一点）

# ============================================================
# 1. Saha 模块
# ============================================================
def total_neutral_density_cm3(P_bar, Tg):
    P = P_bar * 1e5
    n_m3 = P / (kB * Tg)
    return n_m3 / 1e6

C_SAHA = 2.414e21
def saha_ratio(Te_eV, Ei_eV):
    return C_SAHA * (Te_eV ** 1.5) * np.exp(-Ei_eV / Te_eV)

def solve_saha(Te_eV, n_tot_cm3):
    S1, S2, S3, S4 = [saha_ratio(Te_eV, E) for E in E_ION]
    ne = max(1e12, 1e-4 * n_tot_cm3)
    for _ in range(200):
        a1, a2, a3, a4 = S1 / ne, S2 / ne, S3 / ne, S4 / ne
        denom = 1 + a1 + a1*a2 + a1*a2*a3 + a1*a2*a3*a4
        n0 = n_tot_cm3 / denom
        n1 = a1 * n0
        n2 = a1 * a2 * n0
        n3 = a1 * a2 * a3 * n0
        n4 = a1 * a2 * a3 * a4 * n0
        ne_new = n1 + 2*n2 + 3*n3 + 4*n4
        if abs(ne_new - ne) / ne < 1e-6:
            break
        ne = 0.5 * (ne + ne_new)
    return n0, n1, n2, n3, n4, ne

# 粗略功率守恒 Te
def k_ion1(Te): return 4e-13 * np.exp(-11.0 / Te)
def k_ion2(Te): return 2e-13 * np.exp(-20.0 / Te)
def k_ion3(Te): return 1.0e-12 * np.exp(-30.0 / Te)
def k_ion4(Te): return 4e-13 * np.exp(-42.0 / Te)
def k_exc(Te):  return 6e-12 * np.exp(-8.0 / Te)

def self_consistent_Te(P_in_Wcm3, Tg, P_bar):
    n_tot = total_neutral_density_cm3(P_bar, Tg)
    Te_grid = np.linspace(0.3, 3.5, 200)
    diff, ne_grid = [], []
    for Te in Te_grid:
        n0, n1, n2, n3, n4, ne = solve_saha(Te, n_tot)
        pow_ion1 = k_ion1(Te) * ne * n0 * E_ION[0] * e
        pow_ion2 = k_ion2(Te) * ne * n1 * E_ION[1] * e
        pow_ion3 = k_ion3(Te) * ne * n2 * E_ION[2] * e
        pow_ion4 = k_ion4(Te) * ne * n3 * E_ION[3] * e
        pow_exc  = k_exc(Te)  * ne * n0 * E_EXC    * e
        pow_total = pow_ion1 + pow_ion2 + pow_ion3 + pow_ion4 + pow_exc
        diff.append(abs(pow_total - P_in_Wcm3))
        ne_grid.append(ne)
    idx = int(np.argmin(diff))
    return Te_grid[idx], ne_grid[idx], n_tot

# ============================================================
# 2. CR + 线展宽
# ============================================================
def cr_population(n_stage_cm3, Te_eV, ne_cm3, E_u_eV, g_u, A_ul,
                  q_ei=1e-8, q_ir=1e-12):
    dE = E_u_eV
    C_ul = ne_cm3 * q_ei
    C_lu = ne_cm3 * q_ei * g_u * np.exp(-dE / Te_eV)
    C_tot = C_ul + C_lu + q_ir * ne_cm3
    LTE_ratio = g_u * np.exp(-E_u_eV / Te_eV)
    return n_stage_cm3 * LTE_ratio * C_tot / (A_ul + C_tot)

def line_widths_nm(wl_nm, dE_eV, Te_eV, Tg_K, ne_cm3, P_bar):
    M_Xe = 131.29 * 1.66054e-27
    wl_m = wl_nm * 1e-9
    sigma_G_m = wl_m * np.sqrt(2*kB*Tg_K/M_Xe) / c / (2*np.sqrt(2*np.log(2)))
    sigma_G_nm = sigma_G_m * 1e9
    A_Stark = 0.2
    gamma_stark = 1e-14 * A_Stark * (ne_cm3/1e16)**(2/3) * Te_eV**(-0.5)
    gamma_vdw   = 0.05 * P_bar
    gamma_L_nm  = gamma_stark + gamma_vdw
    return sigma_G_nm, gamma_L_nm

def gaussian_phi_lambda(lam_m, lam0_m, sigma_m):
    g = np.exp(-0.5*((lam_m-lam0_m)/sigma_m)**2)
    area = np.trapz(g, lam_m)
    return g / np.maximum(area, 1e-300)

def lorentz_phi_lambda(lam_m, lam0_m, gamma_m):
    L = (gamma_m**2)/((lam_m-lam0_m)**2 + gamma_m**2)
    area = np.trapz(L, lam_m)
    return L / np.maximum(area, 1e-300)

def pseudo_voigt_phi_lambda(lam_m, lam0_m, sigma_m, gamma_m):
    G = gaussian_phi_lambda(lam_m, lam0_m, sigma_m)
    L = lorentz_phi_lambda(lam_m, lam0_m, gamma_m)
    eta = 0.5346*gamma_m/(gamma_m + sigma_m*np.sqrt(2*np.log(2)) + 1e-300)
    phi = (1-eta)*G + eta*L
    return phi / np.maximum(np.trapz(phi, lam_m), 1e-300)

def apply_instrument_response_nm(I_nm, fwhm_nm, wav_grid_nm):
    dlam = float(np.mean(np.diff(wav_grid_nm)))
    sigma_pix = fwhm_nm/dlam/2.355
    return gaussian_filter1d(I_nm, sigma_pix)

# ============================================================
# 3. 连续谱（ff/fb + Xe2*）
# ============================================================
def gaunt_ff_quantum(TK, ne_m3, Z, nu_hz):
    u = h*nu_hz/(kB*TK)
    return 1.0 + 0.217*u**2/(1 + 0.667*u)

def eps_ff_lambda_SI(TK, ne_m3, ni_m3, Z, lam_m):
    nu = c/lam_m
    gff = gaunt_ff_quantum(TK, ne_m3, Z, nu)
    pref = 6.8e-39
    eps_nu = pref*(Z**2)*ne_m3*ni_m3*(TK**-0.5)*np.exp(-h*nu/(kB*TK))*gff

    lam_nm = lam_m * 1e9
    longwave_atten = np.where(lam_nm > 650,
                              np.exp(-(lam_nm - 650) / 100),
                              1.0)
    eps_nu *= longwave_atten
    return eps_nu*(c/lam_m**2)

def eps_fb_lambda_SI(TK, ne_m3, ni_m3, Z, lam_m, Eion_eV=12.13):
    nu = c/lam_m
    E = h*nu
    Eth = Eion_eV*e
    out = np.zeros_like(lam_m)
    mask = (E >= Eth)
    if np.any(mask):
        A = 3.0e-39
        eps_nu = np.zeros_like(nu)
        eps_nu[mask] = A*(Z**4)*ne_m3*ni_m3*(TK**-1.5) * \
                       np.exp(-(E[mask]-Eth)/(kB*TK)) / np.maximum(E[mask]/e, 1.0)
        out = eps_nu*(c/lam_m**2)
    return out

def kappa_ff_lambda_SI(TK, ne_m3, ni_m3, Z, lam_m):
    nu = c/lam_m
    gff = gaunt_ff_quantum(TK, ne_m3, Z, nu)
    pref = (4*e**6)/(3*math.sqrt(3)*(4*math.pi*epsilon_0)**3*m_e*c)
    pref *= math.sqrt(2*math.pi/(3*m_e*kB))
    kappa_nu = pref*(Z**2)*ne_m3*ni_m3*(TK**-1.5)*nu**(-3)*gff
    return kappa_nu*(c/lam_m**2)

def planck_B_lambda(lam_m, TK):
    a = 2.0*h*c**2/lam_m**5
    x = (h*c)/(lam_m*kB*TK)
    return a/np.expm1(x)

def xe2_excimer_profile_lambda(lam_m):
    lam_nm = lam_m * 1e9
    g1 = 0.40 * np.exp(-1.5 * ((lam_nm - 365.0) / 120.0) ** 2)
    g2 = 0.5  * np.exp(-0.5 * ((lam_nm - 525.0) / 52.0) ** 2)
    phi = g1 + g2
    area = np.trapz(phi, lam_m)
    return phi / max(area, 1e-300)

# ============================================================
# 4. 多区辐射输运
# ============================================================
def propagate_zones(lam_m, j_sr_zones, kappa_zones, L_m_list):
    I = np.zeros_like(lam_m)
    for j_sr, kap, Lm in zip(j_sr_zones, kappa_zones, L_m_list):
        kap_safe = np.maximum(kap, 1e-30)
        tau = kap_safe * Lm
        S = j_sr / kap_safe
        I = I * np.exp(-tau) + S * (1.0 - np.exp(-tau))
    return I

# ============================================================
# 5. 线谱库
# ============================================================
Line = namedtuple('Line', 'wl stage E_u E_l g_u A_ul')

LINE_DATA = [
    (228.90329, 1, 18.497400, 13.057189, 1, 3.1094e+07),
    (247.84123, 1, 18.377500, 13.313639, 5, 3.7516e+08),
    (260.69700, 1, 16.076702, 11.266919, 6, 2.3603e+07),

    (450.84048, 1, 18.497400, 15.747332, 5, 2.7570e+07),
    (467.84900, 1, 16.457805, 13.802784, 6, 5.3898e+07),
    (474.65235, 1, 17.653100, 15.024380, 1, 22.265e+08),
    (484.43300, 1, 14.097673, 11.539016, 1, 9.4862e+07),
    (485.37700, 1, 16.356465, 13.802784, 3, 2.9822e+07),
    (486.24500, 1, 16.430240, 13.881133, 4, 5.7752e+07),
    (487.65000, 1, 16.125876, 13.584098, 5, 7.6659e+07),
    (488.35300, 1, 15.080052, 12.541929, 6, 7.0112e+07),
    (488.55781, 1, 17.467300, 14.929541, 1, 3.9951e+07),
    (488.73000, 1, 15.281623, 12.745460, 3, 3.1968e+07),
    (491.96600, 1, 15.444847, 12.925360, 5, 1.6602e+07),
    (492.14800, 1, 15.264009, 12.745460, 6, 8.8097e+07),

    (529.22200, 1, 13.881133, 11.539016, 5, 2.3014e+07),
    (530.92700, 1, 15.080052, 12.745460, 6, 1.1870e+07),
    (531.38700, 1, 16.430240, 14.097673, 1, 5.8379e+07),
    (534.63300, 1, 13.860462, 11.539016, 3, 1.8566e+07),
    (543.89600, 1, 15.024380, 12.745460, 3, 7.2645e+07),

    (765.06600, 1, 14.929541, 13.313639, 6, 1.3653e+07),
    (778.70400, 1, 16.356465, 14.764719, 1, 3.3370e+06),
    (823.57000, 1, 15.080052, 13.584098, 3, 1.8253e+08),
    (828.57000, 1, 15.080052, 13.584098, 3, 5.7253e+07),
    (835.72400, 1, 14.073739, 12.588819, 4, 1.5272e+07),
    (882.46100, 1, 16.391673, 14.983882, 4, 5.2791e+07),
]
LINE_DATA = [Line(*row) for row in LINE_DATA]

# ============================================================
# ✅ 新增：电子能量计算（平均能量 + 能量密度）
# ============================================================
def electron_energy_metrics(Te_eV, ne_cm3):
    eps_e_mean_eV = 1.5 * Te_eV
    ne_m3 = ne_cm3 * 1e6
    u_e_J_m3 = 1.5 * ne_m3 * Te_eV * e
    u_e_J_cm3 = u_e_J_m3 / 1e6
    return eps_e_mean_eV, u_e_J_m3, u_e_J_cm3

# ============================================================
# 6. 单次功率 → 光谱计算（返回 M_lambda 与 Te/ne 等）
# ============================================================
def calc_spectrum_for_power(P_laser_in_W):
    P_in = P_laser_in_W * 40.0  # W -> W/cm^3

    Te_best, ne_best_cm3, n_tot_cm3 = self_consistent_Te(P_in, T_G, P_BAR)

    n0, n1, n2, n3, n4, ne0 = solve_saha(Te_best, n_tot_cm3)

    eps_e_mean_eV, u_e_J_m3, u_e_J_cm3 = electron_energy_metrics(Te_best, ne0)

    ne_m3 = ne0 * 1e6
    n0_m3 = n0 * 1e6
    ni_m3 = (n1 + n2) * 1e6
    TK = T_RAD

    eps_ff_m = eps_ff_lambda_SI(TK, ne_m3, ni_m3, Z=1.0, lam_m=lam_m)
    eps_fb_m = eps_fb_lambda_SI(TK, ne_m3, ni_m3, Z=1.0, lam_m=lam_m)

    phi_exc   = xe2_excimer_profile_lambda(lam_m)
    eps_exc_m = EXC_SCALE * ne_m3 * n0_m3 * phi_exc

    eps_cont_m = (eps_ff_m + eps_fb_m) * FF_SCALE + eps_exc_m
    j_cont_sr  = eps_cont_m / (4.0 * pi)
    j_cont_nm  = j_cont_sr * 1e-9

    kappa_ff_m1 = kappa_ff_lambda_SI(TK, ne_m3, ni_m3, 1.0, lam_m) * KAPPA_CONT_SCALE

    j_cont_sr_zones  = [j_cont_nm.copy() for _ in ZONES]
    kappa_cont_zones = [kappa_ff_m1.copy() for _ in ZONES]

    j_line_sr_zones  = [np.zeros_like(lam_m) for _ in ZONES]
    kappa_line_zones = [np.zeros_like(lam_m) for _ in ZONES]

    B_lam    = planck_B_lambda(lam_m, T_RAD)
    B_lam_nm = B_lam * 1e-9

    zone_ne_cm3 = np.array([ne0 * z['ne_fac'] for z in ZONES])
    zone_Te_eV  = np.array([Te_best * z['Te_fac'] for z in ZONES])
    zones_L_m   = np.array([z['L_m'] for z in ZONES])

    for ln in LINE_DATA:
        if ln.stage == 0:
            n_stage_cm3 = n0
        elif ln.stage == 1:
            n_stage_cm3 = 0.15 * n1
        elif ln.stage == 2:
            n_stage_cm3 = n2
        else:
            continue

        lam0_m = ln.wl * 1e-9
        dE_eV  = ln.E_u - ln.E_l
        hnu    = dE_eV * e

        for zi, (Te_eV_z, ne_cm3_z) in enumerate(zip(zone_Te_eV, zone_ne_cm3)):
            n_u_cm3 = cr_population(n_stage_cm3, Te_eV_z, ne_cm3_z,
                                    ln.E_u, ln.g_u, ln.A_ul)
            n_u_m3  = n_u_cm3 * 1e6

            sigma_nm, gamma_nm = line_widths_nm(ln.wl, dE_eV, Te_eV_z, T_G, ne_cm3_z, P_BAR)
            sigma_m = max(float(sigma_nm), 1e-6) * 1e-9
            gamma_m = max(float(gamma_nm), 1e-6) * 1e-9
            phi = pseudo_voigt_phi_lambda(lam_m, lam0_m, sigma_m, gamma_m)

            j_line_m  = (n_u_m3 * ln.A_ul * hnu / (4.0 * pi)) * phi
            j_line_nm = j_line_m * 1e-9 * LINE_SCALE
            j_line_sr_zones[zi] += j_line_nm

            k_line = ALPHA_LINE * (j_line_nm * 4.0 * pi) / np.maximum(B_lam_nm, 1e-50)
            kappa_line_zones[zi] += k_line

    j_tot_zones     = [j_c + j_l for j_c, j_l in zip(j_cont_sr_zones, j_line_sr_zones)]
    kappa_tot_zones = [k_c + k_l for k_c, k_l in zip(kappa_cont_zones, kappa_line_zones)]
    I_lambda_sr_nm  = propagate_zones(lam_m, j_tot_zones, kappa_tot_zones, zones_L_m)

    I_conv   = apply_instrument_response_nm(I_lambda_sr_nm, RES_FWHM_NM, wav_nm)
    M_lambda = pi * I_conv

    band_power = float(np.trapz(M_lambda, lam_m))
    mean_tau   = float(np.mean(kappa_tot_zones[1] * ZONES[1]['L_m']))

    return M_lambda, Te_best, ne0, eps_e_mean_eV, u_e_J_m3, u_e_J_cm3, band_power, mean_tau

# ============================================================
# 7. 主流程：多功率计算 + 同图叠加 + ✅按气压+功率命名保存txt
# ============================================================
def main():
    plt.figure(figsize=(10, 4.6))

    summary = []  # 保存汇总（用于CSV）
    power_list = P_LASER_LIST

    for P in power_list:
        (M_lambda, Te_best, ne0,
         eps_e_mean_eV, u_e_J_m3, u_e_J_cm3,
         band_power, mean_tau) = calc_spectrum_for_power(P)

        # ✅汇总行
        summary.append([P, Te_best, ne0, eps_e_mean_eV, u_e_J_m3, u_e_J_cm3, band_power, mean_tau])

        # ✅保存光谱数据（按你要求：Xe_OES_abs_{P_BAR:.1f}bar_{int(P)}W.txt）
        out_txt = f"Xe_OES_abs_{P_BAR:.1f}bar_{int(P)}W.txt"
        np.savetxt(
            out_txt,
            np.column_stack([wav_nm, M_lambda]),
            header="wavelength_nm  M_lambda_abs(model_units)",
            comments=''
        )
        print(f"[Saved spectrum] {out_txt}")

        # 叠加绘图
        plt.plot(wav_nm, M_lambda, lw=1.3, label=f'{int(P)} W')

        print(
            f"[{int(P):>4} W] "
            f"Te≈{Te_best:.3f} eV, "
            f"ne≈{ne0:.3e} cm^-3, "
            f"<εe>≈{eps_e_mean_eV:.3f} eV/e, "
            f"u_e≈{u_e_J_m3:.3e} J/m^3 ({u_e_J_cm3:.3e} J/cm^3), "
            f"BandPower≈{band_power:.3e}, "
            f"meanτ≈{mean_tau:.3e}"
        )

    # 图形输出
    plt.xlim(*LAM_DISPLAY)
    plt.xlabel('波长 (nm)')
    plt.ylabel('绝对强度 (模型单位)')
    plt.title(f'Xe OES 绝对光强谱：不同激光功率对比 | {P_BAR:.1f} bar')
    plt.grid(True, ls=':', alpha=0.35)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'Xe_OES_multi_power_absolute_{P_BAR:.1f}bar.png', dpi=300)
    plt.show()

    # ✅保存汇总CSV
    header = "P_laser_W,Te_eV,ne_cm^-3,mean_epsilon_e_eV_per_e,u_e_J_per_m3,u_e_J_per_cm3,BandPower,mean_tau"
    np.savetxt(
        "Xe_power_ne_energy_summary.csv",
        np.array(summary, dtype=float),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.2f"
    )
    print("已保存汇总：Xe_power_ne_energy_summary.csv")

if __name__ == "__main__":
    main()
