from pathlib import Path
import sys
import numpy as np
import discopy.util as util
import matplotlib.pyplot as plt

minidisk_rmax = 1.0
minidisk_Nr = 30
minidisk_Nq = 20

def collapse(rf, data):

    Npl, Nr, Nq = data.shape

    Nr2 = (Nr+1)//2

    rf2 = np.empty(Nr2+1)
    rf2[:-1] = rf[:-1:2]
    rf2[-1] = rf[-1]

    data2 = np.empty((Npl, Nr2, Nq))

    for i in range(Nr2):
        a = 2*i
        b = min(2*i+2, Nr)
        data2[:, i, :] = data[:, a:b, :].sum(axis=1)

    return rf2, data2

def makeSummPlot(rf, data, figname):
    
    a = rf[:-1]
    b = rf[1:]
    r =  2.0/3.0 *  (a*a + a*b + b*b) / (a + b)

    dV = data[:, :, 0]
    dM = data[:, :, 1]
    dMv = data[:, :, 2]
    dj = data[:, :, 3]
    djv = data[:, :, 4]
    dmex = data[:, :, 5]
    dmey = data[:, :, 6]
    dmcos = data[:, :, 16]
    dmsin = data[:, :, 17]
    
    Sig = dM / dV
    vr = dMv / dM
    Mdot = -2*np.pi*r * vr * Sig
    l = dj / dM
    om = l / (r*r)
    Jdot_adv = -2*np.pi*r * djv / dV

    ex = dmex / dM
    ey = dmey / dM
    e = np.sqrt(ex*ex + ey*ey)
    phip = np.arctan2(ey, ex)

    Sig1 = np.sqrt((dmcos / dM) ** 2 + (dmsin / dM) ** 2)
    phip1 = np.arctan2(dmsin, dmcos)

    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    ax[0, 0].plot(r, Sig[0])
    ax[0, 0].plot(r, Sig[1])
    ax[0, 1].plot(r, Sig1[0])
    ax[0, 1].plot(r, Sig1[1])
    ax[0, 2].plot(r, vr[0])
    ax[0, 2].plot(r, vr[1])
    ax[0, 3].plot(r, Mdot[0])
    ax[0, 3].plot(r, Mdot[1])
    ax[1, 0].plot(r, e[0])
    ax[1, 0].plot(r, e[1])
    ax[1, 1].plot(r, phip[0], lw=1)
    ax[1, 1].plot(r, phip[1], lw=1)
    ax[1, 1].plot(r, phip1[0], color='C0', ls='--', lw=2)
    ax[1, 1].plot(r, phip1[1], color='C1', ls='--', lw=2)
    ax[1, 2].plot(r, om[0])
    ax[1, 2].plot(r, om[1])
    ax[1, 3].plot(r, Jdot_adv[0])
    ax[1, 3].plot(r, Jdot_adv[1])

    ax[0, 0].set(ylabel=r'$\Sigma$')
    ax[0, 1].set(ylabel=r'$\Sigma_1 / \Sigma_0$')
    ax[0, 2].set(ylabel=r'$v^r$')
    ax[0, 3].set(ylabel=r'$\dot{M}$')
    ax[1, 0].set(ylabel=r'$e$')
    ax[1, 1].set(ylabel=r'$\phi_p$')
    ax[1, 2].set(ylabel=r'$\Omega$')
    ax[1, 3].set(ylabel=r'$\dot{J}_{\mathrm{adv}}$')

    fig.tight_layout()

    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)

def makeTimeseriesPlot(t, data, figname):
    
    data_tot = data.sum(axis=2)

    to = t / (2*np.pi)

    V = data_tot[:, :, 0]
    M = data_tot[:, :, 1]
    J = data_tot[:, :, 3]
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].plot(to, V[:, 0])
    ax[0].plot(to, V[:, 1])
    ax[1].plot(to, M[:, 0])
    ax[1].plot(to, M[:, 1])
    ax[2].plot(to, J[:, 0])
    ax[2].plot(to, J[:, 1])

    ax[0].set(ylabel=r'$V$', xlabel=r'$t$ (orb)')
    ax[1].set(ylabel=r'$M$', xlabel=r'$t$ (orb)')
    ax[2].set(ylabel=r'$J$', xlabel=r'$t$ (orb)')

    fig.tight_layout()

    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)


def analyzeSnap(filename):

    t, Qarr, _, _, planetDat = util.loadSnapshotArr(filename)

    data = Qarr.reshape((2, minidisk_Nr, minidisk_Nq))

    r_e = np.linspace(0.0, minidisk_rmax, minidisk_Nr+1)

    name = (filename.stem).split("_")[-1]
    figname = "snap_summary_{0:s}.png".format(name)

    makeSummPlot(*(collapse(r_e, data)), figname)

    return t, data


if __name__ == "__main__":

    snapNames = [Path(x) for x in sys.argv[1:]]

    Nt = len(snapNames)

    data = np.empty((Nt, 2, minidisk_Nr, minidisk_Nq))
    t = np.empty(Nt)

    for i, name in enumerate(snapNames):
        t[i], data[i] = analyzeSnap(name)

    dat = data.mean(axis=0)

    r_e = np.linspace(0.0, minidisk_rmax, minidisk_Nr+1)
    a = r_e[:-1]
    b = r_e[1:]
    r =  2.0/3.0 *  (a*a + a*b + b*b) / (a + b)

    figname = "snap_summary_avg.png".format(name)

    makeSummPlot(*(collapse(r_e, dat)), figname)

    makeTimeseriesPlot(t, data, "snap_summary_t.png")



