from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

def addSeries(axes, t, vals, **kwargs):

    for i, ax in enumerate(axes):
        l = ax.plot(t, vals[i], **kwargs)

    return l[0]

if __name__ == "__main__":


    filenames = [Path(x) for x in sys.argv[1:]]

    tmin = 290.0
    tmax = 300.0

    fig, ax = plt.subplots(2, 4, figsize=(15, 9))

    axes = [ax[0, 0], ax[0, 1], ax[0, 2], ax[0, 3], ax[1, 0], ax[1, 1],
            ax[1, 2]]

    handles = []

    for filename in filenames:
        data = np.loadtxt(filename, unpack=True)
        name = filename.stem

        t = data[0]

        idx = (t <= tmax) & (t >= tmin)
        t = t[idx]

        vals = [datum[idx] for datum in data[1:-2]]
        vals.append(data[-2][idx] + data[-1][idx])

        l = addSeries(axes, t, vals, label=name)
        handles.append(l)

    ax[0, 0].set(xlabel=r'$t$ (orbits)', ylabel=r'$T_g$')
    ax[0, 1].set(xlabel=r'$t$ (orbits)', ylabel=r'$T_g(r>a)$')
    ax[0, 2].set(xlabel=r'$t$ (orbits)', ylabel=r'$\Psi^1_R$')
    ax[0, 3].set(xlabel=r'$t$ (orbits)', ylabel=r'$\Psi^1_I$')

    ax[1, 0].set(xlabel=r'$t$ (orbits)', ylabel=r'$e_x$')
    ax[1, 1].set(xlabel=r'$t$ (orbits)', ylabel=r'$e_y$')
    ax[1, 2].set(xlabel=r'$t$ (orbits)', ylabel=r'$\dot{M}$')

    ax[1, 3].legend(handles=handles)

    fig.tight_layout()

    figname = "cb_comp_timeseries.png"
    print("Saving", figname)
    fig.savefig(figname)


