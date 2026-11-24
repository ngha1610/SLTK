# =====================================================================
# HỒI QUY TUYẾN TÍNH ĐƠN - DỮ LIỆU GIÁ NHÀ ĐÀI BẮC
# Biến phụ thuộc Y : house price of unit area
# Phần 1: các bước chung (hàm dùng lại cho mọi biến độc lập)
# Phần 2: ứng dụng - gọi hàm cho từng biến X cần phân tích
# Công thức tự cài bằng phép toán cơ bản; scipy chỉ lấy phân vị t, F, chi2,
# p-value và hàm Phi; statsmodels chỉ dùng để đối chiếu.
#
# Các đoạn đánh dấu [BỔ SUNG] là phần thêm mới; các hàm gốc giữ nguyên.
# =====================================================================
import math
import numpy as np                      # [BỔ SUNG] dùng cho chan_doan
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =====================================================================
# PHẦN 1. CÁC BƯỚC CHUNG
# =====================================================================
FILE = r"https://raw.githubusercontent.com/ngha1610/SLTK/refs/heads/main/Real%20estate%20valuation%20data%20set.csv"   # đường dẫn tới file dữ liệu
ALPHA = 0.05                                    # mức ý nghĩa 5% (tin cậy 95%)
COT_Y = "Y house price of unit area"            # đơn vị: 10000 NT$/ping
VE_DO_THI = True                                # [BỔ SUNG] False nếu không muốn hiện/lưu hình

# Các biến độc lập: số thứ tự -> (tên cột, tên gọi, đơn vị)
BIEN_X = {
    1: ("X1 transaction date",                    "Ngày giao dịch",       "năm"),
    2: ("X2 house age",                           "Tuổi nhà",             "năm"),
    3: ("X3 distance to the nearest MRT station", "Khoảng cách đến MRT",  "m"),
    4: ("X4 number of convenience stores",        "Số cửa hàng tiện lợi", "cửa hàng"),
    5: ("X5 latitude",                            "Vĩ độ",                "độ"),
    6: ("X6 longitude",                           "Kinh độ",              "độ"),
}

# ---------- Bước 0: đọc dữ liệu ----------
df = pd.read_csv(FILE)
df = df.dropna(subset=[COT_Y] + [v[0] for v in BIEN_X.values()])
y = df[COT_Y].tolist()
print(f"Số quan sát n = {len(y)}; mức ý nghĩa alpha = {ALPHA}")


def trung_binh(a):
    """a_bar = (1/n) * tổng(a_i)"""
    return sum(a) / len(a)


def fmt_p(p):
    """p-value rất nhỏ thì in dạng < 0.0001 cho dễ đọc"""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def ket_luan(p, alpha=ALPHA):
    return "Bác bỏ H0" if p < alpha else "Chưa đủ cơ sở bác bỏ H0"


def trung_vi(a):
    s = sorted(a)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


# ---------- [BỔ SUNG] Mục 2.1: kiểm tra dữ liệu (không sửa df) ----------
def kiem_tra_du_lieu():
    """Kiểm tra thiếu, trùng, thống kê mô tả, đếm ngoại lai theo IQR."""
    cot = [COT_Y] + [v[0] for v in BIEN_X.values()]
    print("\n" + "=" * 66)
    print("[2.1] KIỂM TRA DỮ LIỆU")
    print("=" * 66)
    print("n =", len(df))
    print("Giá trị thiếu:", int(df[cot].isna().sum().sum()),
          "; dòng trùng lặp:", int(df.duplicated().sum()))
    print(df[cot].describe().round(3).T.to_string())
    print("Số ngoại lai theo IQR (ngoài [Q1-1.5*IQR; Q3+1.5*IQR]):")
    for c in cot:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        m = (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)
        print(f"  {c}: {int(m.sum())} ngoại lai; max = {df[c].max()}")


# ---------- Bước 1 (4.1.2): bình phương nhỏ nhất ----------
def binh_phuong_nho_nhat(x, y):
    """Tính Sxx, Sxy, b1, b0, phần dư, SSE, SSR, SST."""
    xb, yb = trung_binh(x), trung_binh(y)
    sxx = sum((xi - xb) ** 2 for xi in x)
    sxy = sum((xi - xb) * (yi - yb) for xi, yi in zip(x, y))
    sst = sum((yi - yb) ** 2 for yi in y)            # SST = Syy
    b1 = sxy / sxx
    b0 = yb - b1 * xb
    e = [yi - (b0 + b1 * xi) for xi, yi in zip(x, y)]   # phần dư
    sse = sum(ei ** 2 for ei in e)
    ssr = b1 * sxy                                       # SSR = b1 * Sxy
    return dict(n=len(x), xb=xb, yb=yb, sxx=sxx, sxy=sxy, sst=sst,
                b0=b0, b1=b1, e=e, sse=sse, ssr=ssr)


# ---------- Bước 2 (4.1.3): sigma^2 và sai số chuẩn ----------
def sai_so_chuan(m):
    bac = m["n"] - 2
    s2 = m["sse"] / bac                              # MSE
    se_b1 = math.sqrt(s2 / m["sxx"])
    se_b0 = math.sqrt(s2 * (1 / m["n"] + m["xb"] ** 2 / m["sxx"]))
    return dict(bac=bac, s2=s2, se_b1=se_b1, se_b0=se_b0)


# ---------- Bước 3 (4.1.4): kiểm định t và bảng ANOVA ----------
def kiem_dinh(m, s, alpha=ALPHA):
    t_crit = stats.t.ppf(1 - alpha / 2, s["bac"])    # t_{alpha/2}(n-2)
    t1, t0 = m["b1"] / s["se_b1"], m["b0"] / s["se_b0"]
    p1 = 2 * (1 - stats.t.cdf(abs(t1), s["bac"]))
    p0 = 2 * (1 - stats.t.cdf(abs(t0), s["bac"]))
    msr, mse = m["ssr"] / 1, m["sse"] / s["bac"]
    F = msr / mse
    f_crit = stats.f.ppf(1 - alpha, 1, s["bac"])
    p_F = 1 - stats.f.cdf(F, 1, s["bac"])
    return dict(t_crit=t_crit, t1=t1, t0=t0, p1=p1, p0=p0,
                msr=msr, mse=mse, F=F, f_crit=f_crit, p_F=p_F)


# ---------- Bước 4 (4.1.5): khoảng tin cậy và khoảng dự đoán ----------
def khoang_tin_cay(m, s, kd, x0, alpha=ALPHA):
    t = kd["t_crit"]
    ci_b1 = (m["b1"] - t * s["se_b1"], m["b1"] + t * s["se_b1"])
    ci_b0 = (m["b0"] - t * s["se_b0"], m["b0"] + t * s["se_b0"])
    y0 = m["b0"] + m["b1"] * x0
    h = 1 / m["n"] + (x0 - m["xb"]) ** 2 / m["sxx"]
    se_mean = math.sqrt(s["s2"] * h)                 # cho kỳ vọng mu_{Y|x0}
    se_pred = math.sqrt(s["s2"] * (1 + h))           # cho giá trị y0
    return dict(ci_b1=ci_b1, ci_b0=ci_b0, x0=x0, y0=y0,
                ci_mean=(y0 - t * se_mean, y0 + t * se_mean),
                pi_y0=(y0 - t * se_pred, y0 + t * se_pred))


# ---------- Bước 5: hệ số xác định ----------
def he_so_xac_dinh(m):
    return m["ssr"] / m["sst"]


# ---------- [BỔ SUNG] Mục 2.2: kiểm định phù hợp chi-bình phương (Pearson) ----------
def chi_binh_phuong_chuan(k, kq, h=5, nP_min=5, alpha=ALPHA, in_bang=True):
    """H0: phần dư có phân phối chuẩn. Ước lượng mu bằng e_bar, sigma bằng s (n-1)
    nên r = 2. Quyết định theo miền bác bỏ W = (chi2_{alpha; k-3}; +inf)."""
    e = kq["e"]
    n = len(e)
    eb = trung_binh(e)
    s = math.sqrt(sum((v - eb) ** 2 for v in e) / (n - 1))
    lo = math.floor(min(e) / h) * h
    hi = math.ceil(max(e) / h) * h
    bien = [lo + i * h for i in range(int(round((hi - lo) / h)) + 1)]
    lop = [[bien[i], bien[i + 1]] for i in range(len(bien) - 1)]
    lop[0][0], lop[-1][1] = -math.inf, math.inf       # mở rộng hai đầu ra vô cực

    def tinh(a, b):
        ni = sum(1 for v in e if a <= v < b)           # tần số thực nghiệm n_i
        P = stats.norm.cdf((b - eb) / s) - stats.norm.cdf((a - eb) / s)
        return (a, b, ni, P, n * P)

    while True:                                        # gộp lớp có nP_i < nP_min
        bang = [tinh(a, b) for a, b in lop]
        xau = [i for i, r in enumerate(bang) if r[4] < nP_min]
        if not xau or len(lop) <= 4:
            break
        i = xau[0]
        j = i + 1 if i + 1 < len(lop) else i - 1
        a, b = min(i, j), max(i, j)
        lop[a] = [lop[a][0], lop[b][1]]
        del lop[b]

    chi2 = sum((r[2] - r[4]) ** 2 / r[4] for r in bang)   # công thức (44), mẫu số nP_i
    kk, r_tham_so = len(bang), 2
    bac = kk - r_tham_so - 1
    crit = stats.chi2.ppf(1 - alpha, bac)                 # chi2_{alpha; k-r-1}
    bac_bo = bool(chi2 > crit)                            # chi2_0 thuộc W_alpha?

    if in_bang:
        print(f"\n[2.2] Y theo X{k}: e_bar={eb:.4f}; s={s:.4f}; k={kk}; bậc tự do={bac}")
        print(f"{'(x_i; x_i+1)':>16} {'n_i':>5} {'P_i':>8} {'nP_i':>8} {'n_i-nP_i':>9} {'(.)^2/nP_i':>11}")
        for a, b, ni, P, nP in bang:
            print(f"({a:>6.1f}; {b:>6.1f}) {ni:>5d} {P:>8.4f} {nP:>8.2f} "
                  f"{ni - nP:>9.2f} {(ni - nP) ** 2 / nP:>11.2f}")
        print(f"  chi2_0 = {chi2:.4f}; W = ({crit:.4f}; +inf) -> "
              f"{'chi2_0 thuộc W: Bác bỏ H0' if bac_bo else 'chi2_0 không thuộc W: Chưa đủ cơ sở bác bỏ H0'}")
    return dict(k=k, n=n, s=s, kk=kk, bac=bac, chi2=chi2, crit=crit, bac_bo=bac_bo, bang=bang)


def bang_chi_chuan(ds):
    """Bảng tổng hợp kiểm định chuẩn cho nhiều mô hình."""
    b = pd.DataFrame({
        "Mô hình": [f"Y ~ X{r['k']}" for r in ds],
        "s": [r["s"] for r in ds],
        "k": [r["kk"] for r in ds],
        "bậc tự do": [r["bac"] for r in ds],
        "chi2_0": [r["chi2"] for r in ds],
        "chi2 tới hạn": [r["crit"] for r in ds],
        "Kết luận": ["Bác bỏ H0" if r["bac_bo"] else "Chưa bác bỏ H0" for r in ds],
    })
    print("\n" + "=" * 66)
    print("BẢNG KIỂM ĐỊNH CHUẨN (CHI-BÌNH PHƯƠNG) CỦA PHẦN DƯ")
    print("=" * 66)
    print(b.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return b


# ---------- Bước 6: biểu đồ phân tán + đường hồi quy (4.1.1) ----------
def ve_phan_tan(k, m, luu_file=True):
    cot, ten, dv = BIEN_X[k]
    x = df[cot].tolist()
    plt.figure(figsize=(6, 4.5))
    plt.scatter(x, y, s=14, alpha=0.6)
    xs = [min(x), max(x)]
    plt.plot(xs, [m["b0"] + m["b1"] * v for v in xs], color="red")
    plt.title(f"Y theo X{k}: {ten}")
    plt.xlabel(f"X{k} ({dv})")
    plt.ylabel("Giá nhà trên đơn vị diện tích")
    plt.tight_layout()
    if luu_file:
        plt.savefig(f"scatter_X{k}.png", dpi=150)
    plt.show()


# ---------- [BỔ SUNG] Mục 7: chẩn đoán phần dư (histogram, Q-Q, phần dư vs dự đoán, Cook) ----------
def chan_doan(k, kq, ve_do_thi=True):
    """Biểu đồ chẩn đoán phần dư và khoảng cách Cook. Chỉ đọc kq, không sửa."""
    e = np.array(kq["e"])
    yhat = np.array(y) - e
    n, sxx, xb = kq["n"], kq["sxx"], kq["xb"]
    x = np.array(df[BIEN_X[k][0]])
    h = 1 / n + (x - xb) ** 2 / sxx                      # đòn bẩy (leverage)
    cook = e ** 2 / (2 * kq["mse"]) * h / (1 - h) ** 2   # khoảng cách Cook
    if ve_do_thi:
        es = np.sort(e)
        z = [stats.norm.ppf((i - 0.5) / n) for i in range(1, n + 1)]
        q1, q3 = np.percentile(es, [25, 75])
        z1, z3 = stats.norm.ppf([0.25, 0.75])
        sl = (q3 - q1) / (z3 - z1)
        ic = q1 - sl * z1
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].hist(e, bins=25, density=True, alpha=0.7)
        g = np.linspace(e.min(), e.max(), 200)
        ax[0].plot(g, stats.norm.pdf(g, e.mean(), e.std(ddof=1)), color="red")
        ax[0].set_title(f"Histogram phần dư (Y theo X{k})")
        ax[1].scatter(z, es, s=10)
        ax[1].plot([z[0], z[-1]], [ic + sl * z[0], ic + sl * z[-1]], color="red")
        ax[1].set_title("Q-Q plot")
        ax[2].scatter(yhat, e, s=12)
        ax[2].axhline(0, color="red")
        ax[2].set_title("Phần dư theo giá trị dự đoán")
        ax[2].set_xlabel("Giá trị dự đoán")
        plt.tight_layout()
        plt.savefig(f"chandoan_X{k}.png", dpi=150)
        plt.show()
    return dict(k=k, max_cook=float(cook.max()), so_cook_lon=int((cook > 4 / n).sum()),
                e_max=float(e.max()), e_min=float(e.min()))


# ---------- Bước 7: đối chiếu với statsmodels (chỉ để kiểm tra code) ----------
def doi_chieu(k, kq):
    try:
        import numpy as np
        import statsmodels.api as sm
    except ImportError:
        print("Chưa cài statsmodels nên bỏ qua bước đối chiếu.")
        return None
    x = df[BIEN_X[k][0]].tolist()
    m = sm.OLS(np.array(y), sm.add_constant(np.array(x))).fit()
    ci = m.conf_int(ALPHA)
    pr = m.get_prediction(np.array([[1.0, kq["x0"]]])).summary_frame(alpha=ALPHA)
    ok = all([
        np.allclose([kq["b0"], kq["b1"]], m.params),
        np.isclose(kq["r2"], m.rsquared),
        np.isclose(kq["F"], m.fvalue),
        np.isclose(kq["p1"], m.pvalues[1]),
        np.allclose([kq["ci_b0"], kq["ci_b1"]], ci),
        np.allclose(kq["ci_mean"], [pr["mean_ci_lower"][0], pr["mean_ci_upper"][0]]),
        np.allclose(kq["pi_y0"], [pr["obs_ci_lower"][0], pr["obs_ci_upper"][0]]),
    ])
    print(f"Đối chiếu statsmodels: {'khớp' if ok else 'KHÔNG khớp'}")
    return ok


# ---------- Hàm tổng: chạy mọi bước cho MỘT biến độc lập ----------
def phan_tich(k, x0=None, ve_do_thi=True, kiem_tra=True):
    """Phân tích hồi quy đơn Y theo X_k. x0: giá trị để tính KTC kỳ vọng và
    khoảng dự đoán (mặc định lấy trung vị của X_k)."""
    cot, ten, dv = BIEN_X[k]
    x = df[cot].tolist()
    if x0 is None:
        x0 = trung_vi(x)
    m = binh_phuong_nho_nhat(x, y)
    s = sai_so_chuan(m)
    kd = kiem_dinh(m, s)
    kt = khoang_tin_cay(m, s, kd, x0)
    r2 = he_so_xac_dinh(m)

    print("\n" + "=" * 66)
    print(f"MÔ HÌNH: Y theo X{k} ({ten}, đơn vị: {dv})")
    print("=" * 66)
    print("[4.1.2] Bình phương nhỏ nhất")
    print(f"  x_bar = {m['xb']:.4f}; y_bar = {m['yb']:.4f}")
    print(f"  Sxx = {m['sxx']:.4f}; Sxy = {m['sxy']:.4f}; SST = {m['sst']:.4f}")
    print(f"  b1 = Sxy/Sxx = {m['b1']:.6f}; b0 = y_bar - b1*x_bar = {m['b0']:.6f}")
    print(f"  Phương trình: y_hat = {m['b0']:.4f} + ({m['b1']:.6f}) * x")
    print(f"  SSE = {m['sse']:.4f}; SSR = {m['ssr']:.4f}; SSR + SSE = {m['ssr'] + m['sse']:.4f}")
    print("[4.1.3] Sai số chuẩn")
    print(f"  sigma^2 = MSE = SSE/(n-2) = {s['s2']:.4f}; sigma = {math.sqrt(s['s2']):.4f}")
    print(f"  se(b1) = {s['se_b1']:.6f}; se(b0) = {s['se_b0']:.6f}")
    print(f"[4.1.4] Kiểm định t, t_(alpha/2)({s['bac']}) = {kd['t_crit']:.4f}")
    print(f"  H0: beta1 = 0: t = {kd['t1']:.4f}; p-value = {fmt_p(kd['p1'])} -> {ket_luan(kd['p1'])}")
    print(f"  H0: beta0 = 0: t = {kd['t0']:.4f}; p-value = {fmt_p(kd['p0'])} -> {ket_luan(kd['p0'])}")
    print("  Bảng ANOVA:")
    print(f"    Hồi quy : SS = {m['ssr']:.4f}; bậc tự do = 1; MS = {kd['msr']:.4f}")
    print(f"    Sai số  : SS = {m['sse']:.4f}; bậc tự do = {s['bac']}; MS = {kd['mse']:.4f}")
    print(f"    Tổng    : SS = {m['sst']:.4f}; bậc tự do = {m['n'] - 1}")
    print(f"    F = {kd['F']:.4f}; F_crit = {kd['f_crit']:.4f}; p-value = {fmt_p(kd['p_F'])}"
          f" -> {ket_luan(kd['p_F'])}")
    print(f"    Kiểm tra t1^2 = {kd['t1'] ** 2:.4f} (bằng F)")
    print(f"[4.1.5] Khoảng tin cậy {100 * (1 - ALPHA):.0f}%")
    print(f"  beta1: ({kt['ci_b1'][0]:.6f}; {kt['ci_b1'][1]:.6f})")
    print(f"  beta0: ({kt['ci_b0'][0]:.4f}; {kt['ci_b0'][1]:.4f})")
    print(f"  Tại x0 = {x0:.4f}: y_hat0 = {kt['y0']:.4f}")
    print(f"  KTC cho kỳ vọng mu_(Y|x0): ({kt['ci_mean'][0]:.4f}; {kt['ci_mean'][1]:.4f})")
    print(f"  Khoảng dự đoán cho y0    : ({kt['pi_y0'][0]:.4f}; {kt['pi_y0'][1]:.4f})")
    print("[Hệ số xác định]")
    print(f"  R^2 = SSR/SST = {r2:.4f}  ({100 * r2:.2f}% biến động của Y được X{k} giải thích)")

    kq = {**m, **s, **kd, **kt, "r2": r2, "k": k}
    if ve_do_thi:
        ve_phan_tan(k, m)
    if kiem_tra:
        doi_chieu(k, kq)
    return kq


def bang_tong_hop(danh_sach_kq):
    """Bảng so sánh các mô hình đã phân tích (truyền vào list kết quả)."""
    bang = pd.DataFrame({
        "Mô hình": [f"Y ~ X{r['k']}" for r in danh_sach_kq],
        "b0": [r["b0"] for r in danh_sach_kq],
        "b1": [r["b1"] for r in danh_sach_kq],
        "t (b1)": [r["t1"] for r in danh_sach_kq],
        "p-value (b1)": [fmt_p(r["p1"]) for r in danh_sach_kq],
        "R^2": [r["r2"] for r in danh_sach_kq],
    })
    print("\n" + "=" * 66)
    print("BẢNG TỔNG HỢP")
    print("=" * 66)
    print(bang.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return bang


# =====================================================================
# PHẦN 2. ỨNG DỤNG CHO TỪNG BIẾN ĐỘC LẬP
# =====================================================================
kiem_tra_du_lieu()                                    # [BỔ SUNG] mục 2.1

kq_X1 = phan_tich(1, ve_do_thi=VE_DO_THI)             # Ngày giao dịch
kq_X2 = phan_tich(2, ve_do_thi=VE_DO_THI)             # Tuổi nhà (x0 mặc định = trung vị)
kq_X3 = phan_tich(3, x0=500, ve_do_thi=VE_DO_THI)     # Khoảng cách đến MRT, tại x0 = 500 m
kq_X4 = phan_tich(4, ve_do_thi=VE_DO_THI)             # Số cửa hàng tiện lợi
kq_X5 = phan_tich(5, ve_do_thi=VE_DO_THI)             # Vĩ độ
kq_X6 = phan_tich(6, ve_do_thi=VE_DO_THI)             # Kinh độ
ds_kq = [kq_X1, kq_X2, kq_X3, kq_X4, kq_X5, kq_X6]

bang_tong_hop(ds_kq)                                  # bảng so sánh 6 mô hình

# [BỔ SUNG] Mục 2.2: kiểm định phù hợp chi-bình phương cho phần dư
chi_chuan = [chi_binh_phuong_chuan(r["k"], r, in_bang=True) for r in ds_kq]
bang_chi_chuan(chi_chuan)
# Kiểm tra độ nhạy theo cách chia lớp (lớp rộng 10):
chi_chuan_h10 = [chi_binh_phuong_chuan(r["k"], r, h=10, in_bang=False) for r in ds_kq]
bang_chi_chuan(chi_chuan_h10)

# [BỔ SUNG] Mục 7: chẩn đoán phần dư (histogram, Q-Q, phần dư vs dự đoán, Cook)
chan_doan_kq = [chan_doan(r["k"], r, ve_do_thi=VE_DO_THI) for r in ds_kq]
print("\n" + "=" * 66)
print("CHẨN ĐOÁN PHẦN DƯ: khoảng cách Cook")
print("=" * 66)
print(pd.DataFrame(chan_doan_kq).round(4).to_string(index=False))
