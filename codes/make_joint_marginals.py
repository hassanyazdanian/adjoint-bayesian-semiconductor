"""Stage-A-only marginal posteriors of c_p and c_n for the joint reconstruction figure (Section 6.4).
Reads the four chain files of results/nuts_Setting2_Ex1_prod_v3_joint_A and checks that the draws
reproduce the posterior summaries of production_v3_report/joint/tables.json before plotting.
Style as plotting.plot_histogram (figsize, fonts, colours), but without the KDE, which would smooth
across the hard bounds of the uniform prior, and with the x-axis on the prior interval.
Usage: python3 -I make_joint_marginals.py <run_dir> <tables.json> <out_dir>"""
import hashlib, io, json, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                            "mathtext.fontset": "stix"})

class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] in ('numpy', 'builtins', 'collections', 'copyreg', '_codecs'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

run, tables, out = sys.argv[1:4]
chains, shas = [], []
for i in range(4):
    b = open(os.path.join(run, f"chain{i}.pickle"), "rb").read()
    shas.append(hashlib.sha256(b).hexdigest()); chains.append(Safe(io.BytesIO(b)).load())
J = json.load(open(tables)); A = [s for s in J['stages'] if s['stage'] == 'joint_A'][0]
assert all(c['plateau_bounds'] == {'cp': (-2.5, -1.5), 'cn': (0.5, 1.5)} for c in chains)
assert [c['info']['seed'] for c in chains] == [100, 101, 102, 103]
TEX = {'cp': r'$c_{\mathrm{p}}$', 'cn': r'$c_{\mathrm{n}}$'}
for q, key in (('cp', 'c_p'), ('cn', 'c_n')):
    x = np.concatenate([c[q] for c in chains]); P = A['posterior'][key]
    sd_ok = min(abs(x.std(ddof=d) - P['sd']) for d in (0, 1)) < 1e-12
    assert x.size == 8000 and abs(x.mean() - P['mean']) < 1e-12 and sd_ok, (q, x.mean(), P['mean'])
    for p, k in ((5, 'q05'), (50, 'q50'), (95, 'q95')):
        assert abs(np.percentile(x, p) - P[k]) < 1e-9, (q, k, np.percentile(x, p), P[k])
    a, b = A['plateau_bounds'][key]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.hist(x, bins=np.linspace(a, b, 41), density=True, edgecolor="black", linewidth=0.25, color="lightgray")
    ax.axhline(1.0 / (b - a), color="0.35", linestyle=":", linewidth=1.5, label="Prior")
    ax.axvline(x.mean(), color="green", linestyle="-", linewidth=2.0, label="Mean")
    ax.axvline(P['truth'], color="red", linestyle="--", linewidth=1.5, label="True value")
    ax.set_xlim(a, b); ax.set_ylim(0, 2.25)
    ax.set_xlabel(TEX[q], fontsize=16); ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=12, loc="upper right" if q == 'cp' else "upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"joint_hist_{q}.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    print(q, f"n={x.size} mean={x.mean():.4f} sd={x.std(ddof=1):.4f} q05={P['q05']:.4f} q95={P['q95']:.4f}",
          f"bin max density={np.histogram(x, bins=np.linspace(a, b, 41), density=True)[0].max():.2f}")
print("chain file sha256:", shas)
