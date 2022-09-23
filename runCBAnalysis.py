from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import discopy.util as util
import discopy.geom as geom

class CheckpointDiagnostics:

    def __init__(self, N):
        self.N = N
        self.Nr = None
        self.t = np.empty(N)
        self.dt = np.empty(N)
        self.idx = 0


    def addCheckpoint(self, filename):
        pars = util.loadPars(filename)
        opts = util.loadOpts(filename)

        Nr = pars['Num_R']

        t, rf, zf, f, fv, dt, rjmh, zkmh = util.loadFluxR(filename)
        _, rs, zs, s, sg, sv, ss, sc, sd, _, _, _ = util.loadSource(filename)
        _, _, _, diag, _, _ = util.loadDiagRZ(filename)

        NgRa = 2 if (pars['NoBC_Rmin'] == 0) else 0
        NgRb = 2 if (pars['NoBC_Rmax'] == 0) else 0

        a = NgRa
        b = NgRa + Nr

        if NgRa == 0:
            af = a
            ia = 1
        else:
            af = a-1
            ia = 0
        
        bf = b
        
        r = rs[0, a:b]
        rf = rjmh[a:b+1]
        
        if (r <= rf[:-1]).any():
            print("rs is low?")
        if (r >= rf[1:]).any():
            print("rs is high?")
        
        if self.Nr is None:
            self._initialize_arrays(r, rf)

        self.t[self.idx] = t
        self.dt[self.idx] = dt

        self.Sig[:, self.idx] = diag[0, a:b, 0]
        self.j[:, self.idx] = diag[0, a:b, 3]
        
        self.Mdot_c[:, self.idx] = -diag[0, a:b, 1] * self.dA
        self.MdotAbs_c[:, self.idx] = -diag[0, a:b, 2] * self.dA
        self.JdotAdv_c[:, self.idx] = -diag[0, a:b, 4] * self.dA
        self.JdotAdvAbs_c[:, self.idx] = -diag[0, a:b, 5] * self.dA
        self.dTdr_g1_c[:, self.idx] = diag[0, a:b, 9] * self.dV/self.dr
        self.dTdr_g2_c[:, self.idx] = diag[0, a:b, 10] * self.dV/self.dr

        self.Mdot_f[ia:, self.idx] = -f[0, af:bf, 0]
        self.Jdot_f[ia:, self.idx] = -f[0, af:bf, 3]
        self.Jdot_v[ia:, self.idx] = -fv[0, af:bf, 3]
        self.Mdot_s[1:, self.idx] = ss[0, a:b, 0].cumsum()
        self.Jdot_s[1:, self.idx] = ss[0, a:b, 3].cumsum()
        self.Jdot_g[1:, self.idx] = sg[0, a:b, 3].cumsum()

        self.idx += 1

    def _initialize_arrays(self, r, rf):

        Nr = len(r)
        self.Nr = Nr

        self.r = r
        self.rf = rf

        self.dr = rf[1:] - rf[:-1]
        self.dA = 2*np.pi * r
        self.dV = np.pi * (rf[1:] + rf[:-1]) * self.dr

        self.Sig = np.empty((Nr, self.N))
        self.j = np.empty((Nr, self.N))
        
        self.Mdot_f = np.zeros((Nr+1, self.N))
        self.Mdot_s = np.zeros((Nr+1, self.N))
        
        self.Jdot_f = np.zeros((Nr+1, self.N))
        self.Jdot_v = np.zeros((Nr+1, self.N))
        self.Jdot_s = np.zeros((Nr+1, self.N))
        self.Jdot_g = np.zeros((Nr+1, self.N))

        self.Mdot_c = np.empty((Nr, self.N))
        self.MdotAbs_c = np.empty((Nr, self.N))
        self.JdotAdv_c = np.empty((Nr, self.N))
        self.JdotAdvAbs_c = np.empty((Nr, self.N))
        self.dTdr_g1_c = np.empty((Nr, self.N))
        self.dTdr_g2_c = np.empty((Nr, self.N))


def calc_mean_sd(f_t, dt):

    T = dt.sum()

    f = (f_t*dt).sum(axis=-1) / T
    df = np.sqrt((dt * (f_t-f[..., None])**2).sum(axis=-1) / T)

    return f, df

def plot_band(ax, r, f_t, dt, label, color, **kwargs):

    band = kwargs.pop('band') if 'band' in kwargs else True

    f, df = calc_mean_sd(f_t, dt)
    
    if band:
        ax.fill_between(r, f-df, f+df, alpha=0.5, color=color, lw=0)
    ax.plot(r, f, color=color, label=label, **kwargs)


def getdMdJr(checkpoint):
    
    pars = util.loadPars(checkpoint)
    opts = util.loadOpts(checkpoint)
    t, x1, x2, x3, prim, dat = util.loadCheckpoint(checkpoint)

    dV = geom.getDV(dat, opts, pars)

    rho = prim[:, 0]
    om = prim[:, 3]

    x, y, z = geom.getXYZ(x1, x2, x3, opts, pars)
    r = np.sqrt(x*x + y*y)

    dM = rho*dV
    dJ = rho*r*r*om*dV

    x1_imh = dat[0]
    Nr = pars['Num_R']
    NgRa = 2 if pars['NoBC_Rmin'] == 0 else 0

    Mr = np.empty(Nr)
    Jr = np.empty(Nr)

    for i in range(Nr):
        idx = (x1 > x1_imh[NgRa+i]) & (x1 < x1_imh[NgRa+i+1])
        Mr[i] = dM[idx].sum()
        Jr[i] = dJ[idx].sum()

    return t, Mr, Jr


def integrateDiagOverInterval(t, dt, f, tA, tB):

    if len(f.shape) > 1:
        F = np.zeros(f.shape[:-1])
    else:
        F = 0.0

    t1 = np.maximum(t-dt, tA)
    t2 = np.minimum(t, tB)

    Dt = np.maximum(t2-t1, 0.0)

    F = (f * Dt[..., :]).sum(axis=-1)

    return F

def makeCheckpointConsistencyPlots(gas, checkpointA, checkpointB):

    tA, dMrA, dJrA = getdMdJr(checkpointA)
    tB, dMrB, dJrB = getdMdJr(checkpointB)

    dMf = integrateDiagOverInterval(gas.t, gas.dt, gas.Mdot_f, tA, tB)
    dMs = integrateDiagOverInterval(gas.t, gas.dt, gas.Mdot_s, tA, tB)
    dJf = integrateDiagOverInterval(gas.t, gas.dt, gas.Jdot_f, tA, tB)
    dJv = integrateDiagOverInterval(gas.t, gas.dt, gas.Jdot_v, tA, tB)
    dJg = integrateDiagOverInterval(gas.t, gas.dt, gas.Jdot_g, tA, tB)
    dJs = integrateDiagOverInterval(gas.t, gas.dt, gas.Jdot_s, tA, tB)

    dMc = integrateDiagOverInterval(gas.t, gas.dt, gas.Mdot_c, tA, tB)
    dJc = integrateDiagOverInterval(gas.t, gas.dt, gas.JdotAdv_c, tA, tB)
    dJgc = integrateDiagOverInterval(gas.t, gas.dt,
                                     gas.dTdr_g1_c + gas.dTdr_g2_c, tA, tB)

    dMrf = dMf - dMf[0]
    dMrs = dMs - dMs[0]
    dJrf = dJf - dJf[0]
    dJrv = dJv - dJv[0]
    dJrg = dJg - dJg[0]
    dJrs = dJs - dJs[0]
    
    dM = dMf + dMs
    dMr = dMrf + dMrs
    dJ = dJf + dJv + dJg + dJs
    dJr = dJrf + dJrv + dJrg + dJrs

    MrA = dMrA.cumsum()
    MrB = dMrB.cumsum()
    JrA = dJrA.cumsum()
    JrB = dJrB.cumsum()

    rp = gas.rf[1:]
    rm = gas.rf[:-1]
    r = gas.r
    dr = rp - rm
    dt = tB-tA

    dMf_c = ((rp-r) * dMf[:-1] + (r-rm) * dMf[1:]) / dr
    dJf_c = ((rp-r) * dJf[:-1] + (r-rm) * dJf[1:]) / dr

    fig, ax = plt.subplots(2, 7, figsize=(26, 6))

    fig.suptitle("Checkpoint Conservation & Consistency")

    ax[0, 0].set_title("Mass change in annulus")
    ax[0, 0].plot(gas.r, dMrB-dMrA, lw=2, label=r"Checkpoint $\Delta$")
    ax[0, 0].plot(gas.r, dMf[1:]-dMf[:-1], lw=1, label=r'Flux')
    ax[0, 0].plot(gas.r, dMs[1:]-dMs[:-1], lw=1, label=r'Sink')
    ax[0, 0].plot(gas.r, dM[1:]-dM[:-1], 'k', lw=0.5, label=r'Flux + Sink')
    ax[1, 0].plot(gas.r, (dMrB-dMrA) - (dM[1:]-dM[:-1]))

    ax[0, 1].set_title("Rate of change of mass within radius")
    ax[0, 1].plot(gas.rf[1:], (MrB-MrA) / dt, lw=2,
                  label=r"Checkpoint $\Delta$")
    ax[0, 1].plot(gas.rf[1:], dMrf[1:] / dt, lw=1, label=r'Flux')
    ax[0, 1].plot(gas.rf[1:], dMrs[1:] / dt, lw=1, label=r'Sink')
    ax[0, 1].plot(gas.rf[1:], dMr[1:] / dt, 'k', lw=0.5, label=r'Flux + Sink')
    ax[1, 1].plot(gas.rf[1:], (MrB - MrA - dMr[1:]) / dt)

    ax[0, 2].set_title("Local Accretion Rate")
    ax[0, 2].plot(gas.rf, dMf / (tB-tA), color='k', lw=1, label=r'Flux')
    ax[0, 2].plot(gas.r, dMc / (tB-tA), color='grey', lw=1, label=r'Diagnostic')
    ax[1, 2].plot(gas.r, (dMc - dMf_c) / (tB-tA),
                    color='grey', lw=1)

    ax[0, 3].set_title("Angular Momentum change in annulus")
    ax[0, 3].plot(gas.r, dJrB-dJrA, lw=2,
                  label=r"Checkpoint $\Delta$")
    ax[0, 3].plot(gas.r, dJf[1:]-dJf[:-1], lw=1, label=r'Adv. Flux')
    ax[0, 3].plot(gas.r, dJv[1:]-dJv[:-1], lw=1, label=r'Visc. Flux')
    ax[0, 3].plot(gas.r, dJg[1:]-dJg[:-1], lw=1, label=r'Grav. Source')
    ax[0, 3].plot(gas.r, dJs[1:]-dJs[:-1], lw=1, label=r'Sink Source')
    ax[0, 3].plot(gas.r, dJ[1:]-dJ[:-1], 'k', lw=0.5, label=r'Flux + Source')
    ax[1, 3].plot(gas.r, dJrB-dJrA - (dJ[1:]-dJ[:-1]))

    ax[0, 4].set_title("Rate of change of\nAngular Momentum within radius")
    ax[0, 4].plot(gas.rf[1:], (JrB-JrA) / dt, lw=2,
                  label=r"Checkpoint $\Delta$")
    ax[0, 4].plot(gas.rf[1:], dJrf[1:] / dt, lw=1, label=r'Adv. Flux')
    ax[0, 4].plot(gas.rf[1:], dJrv[1:] / dt, lw=1, label=r'Visc. Flux')
    ax[0, 4].plot(gas.rf[1:], dJrg[1:] / dt, lw=1, label=r'Grav. Source')
    ax[0, 4].plot(gas.rf[1:], dJrs[1:] / dt, lw=1, label=r'Sink Source')
    ax[0, 4].plot(gas.rf[1:], dJr[1:] / dt, 'k', lw=0.5, label=r'Flux + Source')
    ax[1, 4].plot(gas.rf[1:], (JrB-JrA - dJr[1:]) / dt)

    ax[0, 5].set_title("Local Advective Torque")
    ax[0, 5].plot(gas.rf, dJf / (tB-tA), color='k', lw=1, label=r'Adv. Flux')
    ax[0, 5].plot(gas.r, dJc / (tB-tA), color='grey', lw=1, label=r'Diagnostic')
    ax[1, 5].plot(gas.r, (dJc - dJf_c) / (tB-tA),
                    color='grey', lw=1)

    ax[0, 6].set_title("Local Gravitational Torque Density")
    ax[0, 6].plot(gas.r, (dJg[1:]-dJg[:-1]) / (dr*(tB-tA)), color='k', lw=1,
                  label=r'Grav. Source')
    ax[0, 6].plot(gas.r, dJgc / (tB-tA), color='grey', lw=1,
                  label=r'Diagnostic')
    ax[1, 6].plot(gas.r, ((dJg[1:]-dJg[:-1])/dr - dJgc) / (tB-tA),
                    color='grey', lw=1)

    ax[0, 0].set_ylabel("Raw Measures")
    ax[1, 0].set_ylabel("Residual")
    for a in ax[1, :]:
        a.set_xlabel(r'$r$')
    for a in ax[0, :]:
        a.legend()

    fig.tight_layout()

    figname = "checkpoint_consistency.pdf"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)


def analyze(reportFile, checkpoints):

    N = len(checkpoints)

    gas = CheckpointDiagnostics(N)

    for i, checkpoint in enumerate(checkpoints):
        print("Loading", checkpoint)
        gas.addCheckpoint(checkpoint)

    makeCheckpointConsistencyPlots(gas, checkpoints[0], checkpoints[-1])
    makeGasSummaryPlot(reportFile, gas, True, "withVar")
    makeGasSummaryPlot(reportFile, gas, False, "noVar")


def makeGasSummaryPlot(report, gas, showVariance, label):

    r = gas.r
    rf = gas.rf

    rmin = rf.min()
    rmax = rf.max()

    dr = rf[1:] - rf[:-1]

    c = ['C{0:d}'.format(i) for i in range(10)]

    fig, ax = plt.subplots(3, 7, figsize=(24, 9))
    for i, axis in enumerate(ax[:, 0]):
        plot_band(axis, r, gas.Sig,
                  gas.dt, r"$\Sigma$", c[0], band=showVariance)
        axis.set(ylabel=r"$\Sigma$")

    for i, axis in enumerate(ax[:, 1]):
        if i < 2:
            plot_band(axis, rf, gas.Mdot_f,
                      gas.dt, r"Flux", c[1], band=showVariance)
            plot_band(axis, rf, gas.Mdot_s,
                      gas.dt, r"Sink", c[2], band=showVariance)
        plot_band(axis, rf, gas.Mdot_s + gas.Mdot_f,
                  gas.dt, r"Total", c[0], band=showVariance)
        axis.set(ylabel=r"$\dot{M}$ - gas on the grid")

    for i, axis in enumerate(ax[:, 2]):
        if i < 2:
            plot_band(axis, rf, gas.Mdot_f,
                      gas.dt, r"Flux", c[1], band=showVariance)
            plot_band(axis, rf, gas.Mdot_s - gas.Mdot_s[-1],
                      gas.dt, r"Sink", c[2], band=showVariance)
        plot_band(axis, rf, gas.Mdot_s + gas.Mdot_f - gas.Mdot_s[-1],
                  gas.dt, r"Total", c[0], band=showVariance)
        axis.set(ylabel=r"$\dot{M}$ - total")

    for i, axis in enumerate(ax[:, 3]):
        if i < 2:
            plot_band(axis, r, np.diff(gas.Mdot_f, axis=0)/dr[:, None],
                      gas.dt, r"Flux", c[1], band=showVariance)
            plot_band(axis, r, np.diff(gas.Mdot_s, axis=0)/dr[:, None],
                      gas.dt, r"Sink", c[2], band=showVariance)
        plot_band(axis, r, np.diff(gas.Mdot_s + gas.Mdot_f, axis=0)/dr[:, None],
                  gas.dt, r"Total", c[0], band=showVariance)
        axis.set(ylabel=r"$d\dot{M}/dr$")
    
    for i, axis in enumerate(ax[:, 4]):
        if i < 2:
            plot_band(axis, rf, gas.Jdot_f,
                      gas.dt, r"Adv. Flux", c[1], band=showVariance)
            plot_band(axis, rf, gas.Jdot_v,
                      gas.dt, r"Visc. Flux", c[2], band=showVariance)
            plot_band(axis, rf, gas.Jdot_g,
                      gas.dt, r"Grav. Source", c[3], band=showVariance)
            plot_band(axis, rf, gas.Jdot_s,
                      gas.dt, r"Sink Source", c[4], band=showVariance)
        plot_band(axis, rf, gas.Jdot_f + gas.Jdot_v + gas.Jdot_g + gas.Jdot_s,
                  gas.dt, r"$\dot{J}$", c[0], band=showVariance)
        axis.set(ylabel=r"$\dot{J}$ - gas on the grid")
    
    for i, axis in enumerate(ax[:, 5]):
        if i < 2:
            plot_band(axis, rf, gas.Jdot_f,
                      gas.dt, r"Adv. Flux", c[1], band=showVariance)
            plot_band(axis, rf, gas.Jdot_v,
                      gas.dt, r"Visc. Flux", c[2], band=showVariance)
            plot_band(axis, rf, gas.Jdot_g - gas.Jdot_g[-1],
                      gas.dt, r"Grav. Source", c[3], band=showVariance)
            plot_band(axis, rf, gas.Jdot_s - gas.Jdot_s[-1],
                      gas.dt, r"Sink Source", c[4], band=showVariance)
        plot_band(axis, rf, gas.Jdot_f + gas.Jdot_v
                + (gas.Jdot_g - gas.Jdot_g[-1]) + (gas.Jdot_s - gas.Jdot_s[-1]),
                  gas.dt, r"$\dot{J}$", c[0], band=showVariance)
        axis.set(ylabel=r"$\dot{J}$ - total")
    
    for i, axis in enumerate(ax[:, 6]):
        if i < 2:
            plot_band(axis, r, np.diff(gas.Jdot_f, axis=0) / dr[:, None],
                      gas.dt, r"Adv. Flux", c[1], band=showVariance)
            plot_band(axis, r, np.diff(gas.Jdot_v, axis=0) / dr[:, None],
                      gas.dt, r"Visc. Flux", c[2], band=showVariance)
            plot_band(axis, r, np.diff(gas.Jdot_g, axis=0) / dr[:, None],
                      gas.dt, r"Grav. Source", c[3], band=showVariance)
            plot_band(axis, r, np.diff(gas.Jdot_s, axis=0) / dr[:, None],
                      gas.dt, r"Sink Source", c[4], band=showVariance)
        plot_band(axis, r, np.diff(gas.Jdot_f + gas.Jdot_v + gas.Jdot_g
                                    + gas.Jdot_s, axis=0) / dr[:, None],
                  gas.dt, r"$\dot{J}$", c[0], band=showVariance)
        axis.set(ylabel=r"$d\dot{J}/dr$")

    for axis in ax[0, :]:
        axis.set(xlim=(rmin, rmax), xscale='linear')
        axis.legend()
    
    for axis in ax[1, :]:
        axis.set(xlim=(0, 10), xscale='linear')
    
    ax[2, 0].set(xlim=(0.1, rmax), xscale='log', yscale='log')
    for axis in ax[2, 1:]:
        axis.set(xlim=(0.1, rmax), xscale='linear', yscale='linear')
    

    fig.tight_layout()

    figname = "gasSummary_{0:s}.pdf".format(label)
    print("Saving", figname)
    fig.savefig(figname)

    figname = "gasSummary_{0:s}.png".format(label)
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)



if __name__ == "__main__":

    reportFile = Path(sys.argv[1])
    filenames = [Path(x) for x in sys.argv[2:]]

    if len(filenames) == 0:
        print("Need a report and some checkpoints!")
        sys.exit()
    
    analyze(reportFile, filenames)

    
