from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import discopy.util as util

class CheckpointDiagnostics:

    def __init__(self, N):
        self.N = N
        self.Nr = None
        self.t = np.empty(N)
        self.dt = np.empty(N)
        self.idx = 0


    def addCheckpoint(self, filename)
        pars = util.loadPars(filename)
        opts = util.loadOpts(filename)

        Nr = pars['Num_R']

        t, rf, zf, f, fv, dt, rjph, zkph = util.loadFluxR(filename)
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
        rf = rf[a:b+1]

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
        self.dTdr_g1_c[:, self.idx = diag[0, a:b, 9] * self.dV/self.dr
        self.dTdr_g2_c[:, self.idx = diag[0, a:b, 10] * self.dV/self.dr

        self.Mdot_f[ia:, self.idx] = -f[0, af:bf, 0]
        self.Jdot_f[ia:, self.idx] = -f[0, af:bf, 3]
        self.Jdot_v[ia:, self.idx] = -f[0, af:bf, 3]
        self.Mdot_s[1:, self.idx] = ss[0, a:b, 0].cumsum()
        self.Jdot_s[1:, self.idx] = ss[0, a:b, 3].cumsum()
        self.Jdot_g[1:, self.idx] = sg[0, a:b, 3].cumsum()

    def _initialize_arrays(self, r, rf):

        self.Nr = len(r)

        self.r = r
        self.rf = rf

        self.dr = rf[1:] - rf[:-1]
        self.dA = 2*np.pi * r
        self.dV = np.pi * (rf[1:] + rf[:-1]) * dr

        self.Sig = np.empty((Nr, N))
        self.j = np.empty((Nr, N))
        
        self.Mdot_f = np.zeros((Nr+1, N))
        self.Mdot_s = np.zeros((Nr+1, N))
        
        self.Jdot_f = np.zeros((Nr+1, N))
        self.Jdot_v = np.zeros((Nr+1, N))
        self.Jdot_s = np.zeros((Nr+1, N))
        self.Jdot_g = np.zeros((Nr+1, N))

        self.Mdot_c = np.empty((Nr, N))
        self.MdotAbs_c = np.empty((Nr, N))
        self.Jdot_c = np.empty((Nr, N))
        self.JdotAbs_c = np.empty((Nr, N))
        self.Jdot_g1_c = np.empty((Nr, N))
        self.Jdot_g2_c = np.empty((Nr, N))


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

def analyze(reportFile, checkpoints):

    N = len(checkpoints)

    gas = CheckpointDiagnostics(N)

    for i, checkpoint in enumerate(checkpoints):
        print("Loading", checkpoint)
        gas.addCheckpoint(checkpoint)

    
    r = gas.r
    rf = gas.rf

    rmin = rf.min()
    rmax = rf.max()

    c = ['C{0:d}'.format(i) for i in range(10)]

    fig, ax = plt.subplots(3, 4, figsize=(24, 9))
    plot_band(ax[0, 0], r, gas.Sig, dt, r"$\Sigma$", c[0],)
    plot_band(ax[1, 0], r, gas.Sig, dt, r"$\Sigma$", c[0],)
    plot_band(ax[2, 0], r, gas.Sig, dt, r"$\Sigma$", c[0],)

    """
    plot_band(ax[0, 1], r, Mdot_in_t, dt, r"$\dot{M}_{\mathrm{in}}$", c[0])
    plot_band(ax[0, 1], r, Mdot_out_t, dt, r"$\dot{M}_{\mathrm{out}}$", c[1])
    plot_band(ax[0, 1], r, Mdot_t, dt, r"$\dot{M}$", c[2])
    plot_band(ax[0, 1], rf, Mdotf_t, dt, r"$\dot{M}_f$", 'grey')
    plot_band(ax[0, 2], rf, Jdot_advf_t, dt, r"$\dot{J}_{\mathrm{adv}, f}$",
              c[0])
    plot_band(ax[0, 2], rf, Jdot_vf_t, dt, r"$\dot{J}_{\mathrm{visc}, f}$",
              c[1])
    plot_band(ax[0, 2], rf, Jdot_gs_t, dt, r"$\dot{J}_{\mathrm{grav}, s}$",
              c[2])
    plot_band(ax[0, 2], rf, Jdot_t, dt, r"$\dot{J}$",
              'grey')
    plot_band(ax[0, 3], r, dTdr_gs_t, dt, r"$dT/dr_{\mathrm{grav}, s}$",
              c[2])
    plot_band(ax[0, 3], r, dTdr_g_t, dt, r"$dT/dr_{\mathrm{grav}, pl}$",
              c[3])
    plot_band(ax[1, 1], r, Mdot_in_t, dt, r"$\dot{M}_{\mathrm{in}}$", c[0])
    plot_band(ax[1, 1], r, Mdot_out_t, dt, r"$\dot{M}_{\mathrm{out}}$", c[1])
    plot_band(ax[1, 1], r, Mdot_t, dt, r"$\dot{M}$", c[2])
    plot_band(ax[1, 1], rf, Mdotf_t, dt, r"$\dot{M}_f$", 'grey')
    plot_band(ax[1, 2], rf, Jdot_advf_t, dt, r"$\dot{J}_{\mathrm{adv}, f}$",
              c[0], band=False)
    plot_band(ax[1, 2], rf, Jdot_vf_t, dt, r"$\dot{J}_{\mathrm{visc}, f}$",
              c[1], band=False)
    plot_band(ax[1, 2], rf, Jdot_gs_t, dt, r"$\dot{J}_{\mathrm{grav}, s}$",
              c[2], band=False)
    plot_band(ax[1, 2], rf, Jdot_t, dt, r"$\dot{J}$",
              'grey', band=False)
    plot_band(ax[1, 3], r, dTdr_gs_t, dt, r"$dT/dr_{\mathrm{grav}, s}$",
              c[2], band=False)
    plot_band(ax[1, 3], r, dTdr_g_t, dt, r"$dT/dr_{\mathrm{grav}, pl}$",
              c[3], band=False)
    """

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[0, 2].legend()
    ax[0, 3].legend()

    ax[0, 0].set(xlim=(rmin, rmax), ylabel=r'$\Sigma$')
    ax[1, 0].set(xlim=(0, 10), xscale='log', yscale='log',
                 ylabel=r'$\Sigma$')
    ax[2, 0].set(xlim=(0.1, rmax), xscale='log', yscale='log',
                 ylabel=r'$\Sigma$')
    """
    ax[0, 1].set(xlim=(rmin, rmax), ylabel=r'$\dot{M}$')
    ax[0, 2].set(xlim=(rmin, rmax), ylabel=r'$\dot{J}$')
    ax[0, 3].set(xlim=(rmin, rmax), ylabel=r'$dT/dr$')
    ax[1, 1].set(xlim=(rjph[1], rmax), xscale='log', yscale='log',
                 ylabel=r'$\dot{M}$')
    ax[1, 2].set(xlim=(rjph[1], rmax), xscale='log',
                 ylabel=r'$\dot{J}$')
    ax[1, 3].set(xlim=(rjph[1], rmax), xscale='log',
                 ylabel=r'$dT/dr$')
    """

    fig.tight_layout()

    figname = "summary.pdf"
    print("Saving", figname)
    fig.savefig(figname)

    figname = "summary.png"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)




if __name__ == "__main__":

    filenames = [Path(x) for x in sys.argv[1:]]

    if len(filenames) == 0:
        print("Need some checkpoints!")
        sys.exit()
    
    analyze(filenames)

    
