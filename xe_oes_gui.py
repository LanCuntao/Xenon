# -*- coding: utf-8 -*-
"""
Xe OES GUI (Tkinter + Matplotlib)

- Uses your existing python script as backend (loaded by file path)
- UI similar to your screenshot: inputs + buttons + log area + embedded plot
- Avoids numpy.trapezoid issue (uses np.trapz; backend already uses np.trapz)

Run:
    python xe_oes_gui.py

Package to EXE (optional):
    pip install pyinstaller
    pyinstaller -F -w xe_oes_gui.py
"""

import os
import sys
import threading
import traceback
import importlib.util
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ========== 你需要改这里：指向你的后端脚本 ==========
def resource_path(rel_path: str):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)

BACKEND_PY = resource_path("T_10atm_不同功率_20251221_1.py")

# ================================================


def load_backend(py_path: str):
    """Load backend module from a .py file path."""
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"Backend file not found:\n{py_path}")

    mod_name = "xe_oes_backend_loaded"
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to create module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_float(s: str, name: str):
    try:
        return float(s)
    except Exception:
        raise ValueError(f"{name} must be a number, got: {s}")


def parse_int(s: str, name: str):
    try:
        return int(float(s))
    except Exception:
        raise ValueError(f"{name} must be an integer, got: {s}")


def parse_power_list(s: str):
    """
    '150,300,450' -> [150.0, 300.0, 450.0]
    Also supports spaces / Chinese comma.
    """
    s = s.replace("，", ",").strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


class XeOESApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Xe OES 计算界面（输入参数 -> 输出光谱）")
        self.geometry("1150x720")

        self.backend = None
        self.results = []   # list of dict for last run

        # --- UI layout ---
        self._build_ui()

        # load backend
        try:
            self.backend = load_backend(BACKEND_PY)
            self._log(f"[OK] 已加载后端脚本：{BACKEND_PY}\n")
            self._sync_defaults_from_backend()
        except Exception as e:
            self._log("[ERROR] 后端脚本加载失败。\n")
            self._log(traceback.format_exc() + "\n")
            messagebox.showerror("加载失败", f"后端脚本加载失败：\n{e}\n\n请检查 BACKEND_PY 路径或用“选择后端脚本”按钮。")

    def _build_ui(self):
        # top frame: inputs
        frm_top = ttk.LabelFrame(self, text="输入参数")
        frm_top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        # left inputs
        left = ttk.Frame(frm_top)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 15), pady=8)

        # right display for "显示波段与网格"
        right = ttk.Frame(frm_top)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=8)

        # ---- input vars ----
        self.var_backend_path = tk.StringVar(value=BACKEND_PY)

        self.var_lam_min = tk.StringVar(value="200")
        self.var_lam_max = tk.StringVar(value="1000")
        self.var_npts    = tk.StringVar(value="4001")
        self.var_fwhm    = tk.StringVar(value="1.8")
        self.var_powers  = tk.StringVar(value="200,450,900,1800,2700,3600,4500,5400,6300")

        # optional: pressure & temperatures (backend has defaults)
        self.var_pbar = tk.StringVar(value="10.0")
        self.var_tg   = tk.StringVar(value="9000")
        self.var_trad = tk.StringVar(value="9000")

        # ---- grid placement ----
        r = 0
        ttk.Label(left, text="后端脚本 (py)：").grid(row=r, column=0, sticky="w", pady=4)
        ent_backend = ttk.Entry(left, textvariable=self.var_backend_path, width=62)
        ent_backend.grid(row=r, column=1, sticky="w", pady=4)
        ttk.Button(left, text="选择后端脚本", command=self._pick_backend).grid(row=r, column=2, sticky="w", padx=6)
        r += 1

        ttk.Label(left, text="波长下限 (nm)：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(left, textvariable=self.var_lam_min, width=18).grid(row=r, column=1, sticky="w", pady=4)
        r += 1

        ttk.Label(left, text="波长上限 (nm)：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(left, textvariable=self.var_lam_max, width=18).grid(row=r, column=1, sticky="w", pady=4)
        r += 1

        ttk.Label(left, text="采样点数 (建议 4001)：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(left, textvariable=self.var_npts, width=18).grid(row=r, column=1, sticky="w", pady=4)
        r += 1

        ttk.Label(left, text="仪器函数 FWHM (nm)：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(left, textvariable=self.var_fwhm, width=18).grid(row=r, column=1, sticky="w", pady=4)
        r += 1

        ttk.Label(left, text="功率列表 P_laser_in (W, 逗号分隔)：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(left, textvariable=self.var_powers, width=62).grid(row=r, column=1, sticky="w", pady=4, columnspan=2)
        r += 1

        # optional physics inputs
        ttk.Label(left, text="气压 P (bar)：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(left, textvariable=self.var_pbar, width=18).grid(row=r, column=1, sticky="w", pady=4)
        r += 1

        ttk.Label(left, text="重粒子温度 Tg (K)：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(left, textvariable=self.var_tg, width=18).grid(row=r, column=1, sticky="w", pady=4)
        r += 1

        ttk.Label(left, text="辐射温度 Trad (K)：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(left, textvariable=self.var_trad, width=18).grid(row=r, column=1, sticky="w", pady=4)
        r += 1

        # buttons
        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Button(btn_frame, text="计算并显示 OES", command=self._run_compute_thread).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="保存当前图为 PNG", command=self._save_png).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="退出", command=self.destroy).pack(side=tk.LEFT)

        # right: show band & grid
        ttk.Label(right, text="显示波段与网格", font=("Microsoft YaHei", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

        ttk.Label(right, text="λ_min").grid(row=1, column=0, sticky="w", pady=2)
        self.lab_lmin = ttk.Entry(right, width=12, state="readonly")
        self.lab_lmin.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(right, text="λ_max").grid(row=2, column=0, sticky="w", pady=2)
        self.lab_lmax = ttk.Entry(right, width=12, state="readonly")
        self.lab_lmax.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(right, text="N_pts").grid(row=3, column=0, sticky="w", pady=2)
        self.lab_npts = ttk.Entry(right, width=12, state="readonly")
        self.lab_npts.grid(row=3, column=1, sticky="w", pady=2)

        # main: plot + log
        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # plot
        plot_frame = ttk.Frame(mid)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(9, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        # log
        log_frame = ttk.LabelFrame(mid, text="计算日志 (Te / ne / 积分功率 / 光学厚度)")
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10, 0))

        self.txt_log = tk.Text(log_frame, height=10)
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.txt_log.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log.config(yscrollcommand=scroll.set)

    def _set_readonly_entry(self, entry: ttk.Entry, value: str):
        entry.config(state="normal")
        entry.delete(0, tk.END)
        entry.insert(0, value)
        entry.config(state="readonly")

    def _sync_defaults_from_backend(self):
        """Fill UI with backend current defaults if available."""
        if self.backend is None:
            return
        # these variables exist in your backend script
        try:
            self.var_pbar.set(str(getattr(self.backend, "P_BAR", 10.0)))
            self.var_tg.set(str(getattr(self.backend, "T_G", 9000.0)))
            self.var_trad.set(str(getattr(self.backend, "T_RAD", 9000.0)))

            lam_disp = getattr(self.backend, "LAM_DISPLAY", (200, 1000))
            self.var_lam_min.set(str(lam_disp[0]))
            self.var_lam_max.set(str(lam_disp[1]))

            npts = len(getattr(self.backend, "wav_nm", np.linspace(200, 1000, 4001)))
            self.var_npts.set(str(npts))

            self.var_fwhm.set(str(getattr(self.backend, "RES_FWHM_NM", 1.8)))

            plist = getattr(self.backend, "P_LASER_LIST", [])
            if plist:
                self.var_powers.set(",".join(str(int(p)) if float(p).is_integer() else str(p) for p in plist))
        except Exception:
            pass

        # update right side box
        self._update_band_grid_box()

    def _update_band_grid_box(self):
        self._set_readonly_entry(self.lab_lmin, self.var_lam_min.get())
        self._set_readonly_entry(self.lab_lmax, self.var_lam_max.get())
        self._set_readonly_entry(self.lab_npts, self.var_npts.get())

    def _pick_backend(self):
        path = filedialog.askopenfilename(
            title="选择后端 Python 文件",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if not path:
            return
        self.var_backend_path.set(path)
        try:
            self.backend = load_backend(path)
            self._log(f"[OK] 已加载后端脚本：{path}\n")
            self._sync_defaults_from_backend()
        except Exception as e:
            self._log("[ERROR] 后端脚本加载失败。\n")
            self._log(traceback.format_exc() + "\n")
            messagebox.showerror("加载失败", f"后端脚本加载失败：\n{e}")

    def _log(self, s: str):
        self.txt_log.insert(tk.END, s)
        self.txt_log.see(tk.END)
        self.txt_log.update_idletasks()

    def _run_compute_thread(self):
        t = threading.Thread(target=self._run_compute, daemon=True)
        t.start()

    def _run_compute(self):
        try:
            if self.backend is None:
                raise RuntimeError("Backend not loaded.")

            # ---- read inputs ----
            lam_min = parse_float(self.var_lam_min.get(), "λ_min")
            lam_max = parse_float(self.var_lam_max.get(), "λ_max")
            npts = parse_int(self.var_npts.get(), "N_pts")
            fwhm = parse_float(self.var_fwhm.get(), "FWHM")
            pbar = parse_float(self.var_pbar.get(), "P(bar)")
            tg = parse_float(self.var_tg.get(), "Tg")
            trad = parse_float(self.var_trad.get(), "Trad")
            powers = parse_power_list(self.var_powers.get())
            if not powers:
                raise ValueError("功率列表为空。")

            if lam_max <= lam_min:
                raise ValueError("λ_max 必须大于 λ_min。")
            if npts < 200:
                raise ValueError("采样点数太小，建议 >= 2000（常用 4001）。")

            # ---- push settings to backend globals ----
            self.backend.P_BAR = float(pbar)
            self.backend.T_G = float(tg)
            self.backend.T_RAD = float(trad)
            self.backend.LAM_DISPLAY = (float(lam_min), float(lam_max))
            self.backend.RES_FWHM_NM = float(fwhm)
            self.backend.P_LASER_LIST = [float(p) for p in powers]

            # rebuild wavelength grid in backend (IMPORTANT)
            self.backend.wav_nm = np.linspace(lam_min, lam_max, npts)
            self.backend.lam_m = self.backend.wav_nm * 1e-9

            # update right panel
            self._update_band_grid_box()

            # ---- compute & plot ----
            self._log("\n============================\n")
            self._log("[Start] 计算 OES ...\n")

            self.results = []
            wav_nm = self.backend.wav_nm

            # clear plot
            self.ax.clear()
            self.ax.set_xlabel("Wavelength (nm)")
            self.ax.set_ylabel("Absolute intensity (model units)")
            self.ax.grid(True, alpha=0.3)

            for P in powers:
                (M_lambda, Te_best, ne0,
                 eps_e_mean_eV, u_e_J_m3, u_e_J_cm3,
                 band_power, mean_tau) = self.backend.calc_spectrum_for_power(P)

                self.results.append({
                    "P_W": P,
                    "Te_eV": Te_best,
                    "ne_cm3": ne0,
                    "eps_mean_eV": eps_e_mean_eV,
                    "u_e_J_m3": u_e_J_m3,
                    "u_e_J_cm3": u_e_J_cm3,
                    "band_power": band_power,
                    "mean_tau": mean_tau,
                    "wav_nm": wav_nm.copy(),
                    "M_lambda": np.array(M_lambda, dtype=float),
                })

                # plot line
                self.ax.plot(wav_nm, M_lambda, lw=1.2, label=f"{int(P)} W" if float(P).is_integer() else f"{P:g} W")

                # log
                self._log(
                    f"[{P:>7g} W] "
                    f"Te≈{Te_best:.3f} eV, "
                    f"ne≈{ne0:.3e} cm^-3, "
                    f"<εe>≈{eps_e_mean_eV:.3f} eV/e, "
                    f"u_e≈{u_e_J_m3:.3e} J/m^3 ({u_e_J_cm3:.3e} J/cm^3), "
                    f"BandPower≈{band_power:.3e}, "
                    f"meanτ≈{mean_tau:.3e}\n"
                )

            self.ax.set_xlim(lam_min, lam_max)
            self.ax.set_title(f"Xe OES: multi-power comparison | {pbar:.1f} bar", fontsize=12, weight="bold")
            self.ax.legend(ncol=3, fontsize=9, frameon=False, loc="upper right")

            self.canvas.draw()
            self._log("[OK] 计算完成。\n")

        except Exception as e:
            self._log("[ERROR] 运行失败：\n")
            self._log(traceback.format_exc() + "\n")
            messagebox.showerror("运行失败", str(e))

    def _save_png(self):
        if self.fig is None:
            return
        path = filedialog.asksaveasfilename(
            title="保存 PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            self._log(f"[Saved] PNG: {path}\n")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))


if __name__ == "__main__":
    app = XeOESApp()
    app.mainloop()
