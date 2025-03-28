import numpy as np
import matplotlib.pyplot as plt
import json

num_matvecss = [50, 200, 400, 900]
names = {
    "ortho_mgs": "MGS",
    "ortho_trunc": "IOP",
    "ortho_cgs": "CGS",
    "ortho_cgs_projreortho": "CGS w/ reo.",
    "ortho_cgs_reortho": "CGS w/ proj.",
}

data = json.load(open(f"../outputs/cpu_benchmark_grcar.json"))
fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
width = 1 / 5
multiplier = 0
colors = plt.color_sequences["tab10"]
for n in num_matvecss:
    base = np.median(data["ortho_mgs"][str(n)]["ts"])
    for m2, method in enumerate(['ortho_trunc', 'ortho_cgs', 'ortho_cgs_projreortho', 'ortho_cgs_reortho']):
        offset = width * multiplier
        rects = ax.bar(offset, np.median(data[method][str(n)]["ts"] / base), width,
                       label=names[method] if n == 50 else "",
                       color=colors[m2 + 1])
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    multiplier += 1
ax.set_ylabel('Runtime relative to MGS')
ax.set_xticks(np.arange(len(num_matvecss)) + 0.5 - width, num_matvecss)
ax.legend(loc='upper right', ncols=2)
ax.set_xlabel("Number of Arnoldi steps")
# ax.set_yscale("log")
fig.tight_layout()
fig.savefig(f"../outputs/cpu_benchmark_grcar_rel.pdf")
plt.show()

data = json.load(open(f"../outputs/cpu_benchmark_grcar.json"))
fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
width = 1/6
multiplier = 0
colors = plt.color_sequences["tab10"]
for n in num_matvecss:
    for m2, method in enumerate(['ortho_mgs', 'ortho_trunc', 'ortho_cgs', 'ortho_cgs_projreortho', 'ortho_cgs_reortho']):
        offset = width * multiplier
        rects = ax.bar(offset, np.median(data[method][str(n)]["ts"]), width, label=names[method] if n == 50 else "", color=colors[m2])
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    multiplier += 1
ax.set_ylabel('Runtime [s]')
ax.set_xticks(np.arange(len(num_matvecss)) + 0.5 - width, num_matvecss)
ax.set_xlabel("Number of Arnoldi steps")
ax.legend(loc='upper left', ncols=3)
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(f"../outputs/cpu_benchmark_grcar.pdf")
plt.show()
#
# data = np.load(f"../outputs/loss_of_orthogonality_grcar.npz")
#
# fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
# ax.plot([i for i in range(1, len(data["ortho_mgs"]) + 1)], data["ortho_mgs"], label="MGS")
# ax.plot([i for i in range(1, len(data["ortho_trunc"]) + 1)], data["ortho_trunc"], label="IOP")
# ax.plot([i for i in range(1, len(data["ortho_cgs"]) + 1)], data["ortho_cgs"], label="CGS")
# ax.plot([i for i in range(1, len(data["ortho_cgs_projreortho"]) + 1)], data["ortho_cgs_projreortho"], label="CGS w/ proj.")
# ax.plot([i for i in range(1, len(data["ortho_cgs_projreortho"]) + 1)], data["ortho_cgs_reortho"], label="CGS w/ reo.")
# ax.set_yscale("log")
# ax.set_xlabel("Number of Arnoldi steps")
# ax.set_ylabel("Loss of orthogonality")
# ax.legend()
# fig.tight_layout()
# fig.savefig(f"../outputs/loss_of_orthogonality_grcar.pdf")
# plt.show()