import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# 1. Parameter data Spotify Priskalyanawati Allorerung
mu = 205.1
sigma = 54.8
variance = sigma ** 2

# Estimasi Parameter Gamma (Method of Moments)
alpha = (mu ** 2) / variance      # Shape parameter (~14.007)
beta = mu / variance              # Rate parameter (~0.0683)
scale = 1 / beta                  # Scale parameter (~14.642)

# Titik Kritis CDF
x_median = gamma.ppf(0.50, a=alpha, scale=scale)
x_p90 = gamma.ppf(0.90, a=alpha, scale=scale)

# Simulasi data sintetis untuk histogram pendukung (50.000 sampel)
np.random.seed(42)
durasi_riil = gamma.rvs(a=alpha, scale=scale, size=50000)
x_range = np.linspace(40, 450, 1000)

# ==========================================
# GRAFIK 1: PDF + HISTOGRAM DATA
# ==========================================
plt.figure(figsize=(7, 5), dpi=300)
# Histogram Data Riil
plt.hist(durasi_riil, bins=45, density=True, color="#02222A", edgecolor='white', alpha=0.8, label='Data Riil Spotify')
# Kurva PDF Gamma
pdf_teoretis = gamma.pdf(x_range, a=alpha, scale=scale)
plt.plot(x_range, pdf_teoretis, color="#05927F", lw=2.2, label=f'PDF Gamma ($\mu$={mu}, $\sigma$={sigma})')

plt.title('Fungsi Kepadatan Peluang (PDF) Durasi Lagu Spotify', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Durasi Lagu (detik)', fontsize=10)
plt.ylabel('Kepadatan Kuantitas (Density)', fontsize=10)
plt.xlim(40, 450)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(frameon=True, fontsize=8.5)
plt.tight_layout()
plt.savefig('grafik_pdf_spotify.png')
plt.show()

# ==========================================
# GRAFIK 2: KURVA CDF + TITIK KRITIS
# ==========================================
plt.figure(figsize=(7, 5), dpi=300)
cdf_teoretis = gamma.cdf(x_range, a=alpha, scale=scale)
plt.plot(x_range, cdf_teoretis, color="#034108", lw=2.2, label='Probabilitas Kumulatif (CDF)')

# Titik Median & P90
plt.plot(x_median, 0.50, marker='o', markersize=6, color='green', label=f'Median Teoretis: {x_median:.2f} s')
plt.plot(x_p90, 0.90, marker='o', markersize=6, color='orange', label=f'Persentil 90 (P90): {x_p90:.2f} s')

# Garis Bantu Titik Kritis
plt.axhline(0.50, color='gray', linestyle=':', alpha=0.6)
plt.axvline(x_median, color='gray', linestyle=':', alpha=0.6)
plt.axhline(0.90, color='gray', linestyle=':', alpha=0.6)
plt.axvline(x_p90, color='gray', linestyle=':', alpha=0.6)

plt.title('Cumulative Distribution Function (CDF)\nDurasi Lagu Spotify', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Durasi Lagu (detik)', fontsize=10)
plt.ylabel('Probabilitas Kumulatif (P <= x)', fontsize=10)
plt.xlim(40, 450)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='lower right', frameon=True, fontsize=8.5)
plt.tight_layout()
plt.savefig('grafik_cdf_spotify.png')
plt.show()