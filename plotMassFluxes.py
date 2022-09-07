from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import discopy.util as util

def getDiagnostics(filename):

    pars = util.loadPars(filename)
    opts = util.loadOpts(filename)

    t, r, phi, z, prim, dat = util.loadCheckpoint(filename)
    _, rf, zf, f, fv, dt, rjph, zkph = util.loadFluxR(filename)
    _, rs, zs, s, sg, sv, ss, sc, sd, _, _, _ = util.loadSource(filename)
    _, _, _, diag, _, _ = util.loadDiagRZ(filename)

    NgRa = 2 if (pars['NoBC_Rmin'] == 0) else 0
    NgRb = 2 if (pars['NoBC_Rmin'] == 0) else 0

    Nr = rjph.shape[0] - 1 - NgRa - NgRb

    a = NgRa
    b = NgRa + Nr

    rs = rs[0, a:b]
    rjph = rjph[a:b+1]
    f = f[0, a:b, :]
    fv = fv[0, a:b, :]
    s = s[0, a:b, :]
    sg = sg[0, a:b, :]
    sv = sv[0, a:b, :]
    ss = ss[0, a:b, :]
    diag = diag[0, a:b, :]

    return dict(t=t, dt=dt, rjph=rjph, rs=rs, f=f, fv=fv,
                s=s, sg=sg, sv=sv, ss=ss, diag=diag)

def calc_mean_sd(f_t, dt):

    T = dt.sum()

    f = (f_t*dt).sum(axis=-1) / T
    df = np.sqrt((dt * (f_t-f[..., None])**2).sum(axis=-1) / T)

    return f, df

def plot_band(ax, r, f_t, dt, label, color, band=True):

    f, df = calc_mean_sd(f_t, dt)
    
    if band:
        ax.fill_between(r, f-df, f+df, alpha=0.5, color=color, lw=0)
    ax.plot(r, f, color=color, label=label)

def analyze(checkpoints):

    N = len(checkpoints)

    for i, checkpoint in enumerate(checkpoints):
        print("Loading", checkpoint)
        diag_dict = getDiagnostics(checkpoint)
        if i == 0:
            rjph = diag_dict['rjph']
            r = diag_dict['rs']

            t = np.empty(N+1)
            dt = np.empty(N)
            f_t = np.empty(diag_dict['f'].shape + (N,))
            fv_t = np.empty(diag_dict['fv'].shape + (N,))
            s_t = np.empty(diag_dict['s'].shape + (N,))
            sv_t = np.empty(diag_dict['sv'].shape + (N,))
            sg_t = np.empty(diag_dict['sg'].shape + (N,))
            ss_t = np.empty(diag_dict['ss'].shape + (N,))
            diag_t = np.empty(diag_dict['diag'].shape + (N,))

            t[0] = diag_dict['t'] - diag_dict['dt']
        t[i] = diag_dict['t']
        dt[i] = diag_dict['dt']
        f_t[..., i] = diag_dict['f']
        fv_t[..., i] = diag_dict['fv']
        s_t[..., i] = diag_dict['s']
        sv_t[..., i] = diag_dict['sv']
        sg_t[..., i] = diag_dict['sg']
        ss_t[..., i] = diag_dict['ss']
        diag_t[..., i] = diag_dict['diag']

    print("Calculating")

    rf = rjph[1:-1]

    rmin = rjph[0]
    rmax = rjph[-1]

    Sig_t = diag_t[:, 0, :]
    Mdot_t = -2*np.pi*r[:, None] * diag_t[:, 1, :]
    MdotAbs_t = -2*np.pi*r[:, None] * diag_t[:, 2, :]
    Mdotf_t = -f_t[:, 0, :]

    Mdot_in_t = 0.5*(Mdot_t - MdotAbs_t)
    Mdot_out_t = -0.5*(Mdot_t + MdotAbs_t)

    Jdot_adv_t = -2*np.pi*r[:, None] * diag_t[:, 4, :]
    Jdot_advAbs_t = -2*np.pi*r[:, None] * diag_t[:, 5, :]

    Jdot_g1_t = -2*np.pi*r[:, None] * diag_t[:, 9, :]
    Jdot_g2_t = -2*np.pi*r[:, None] * diag_t[:, 10, :]
    Jdot_g_t = Jdot_g1_t + Jdot_g2_t

    Jdot_gs_t = -sg_t[:, 3, :]
    Jdot_vs_t = -sv_t[:, 3, :]

    Jdot_advf_t = -f_t[:, 3, :]
    Jdot_vf_t = -fv_t[:, 3, :]

    c = ['C{0:d}'.format(i) for i in range(10)]

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    plot_band(ax[0, 0], r, Sig_t, dt, r"$\Sigma$", c[0],)
    plot_band(ax[0, 1], r, Mdot_in_t, dt, r"$\dot{M}_{\mathrm{in}}$", c[0])
    plot_band(ax[0, 1], r, Mdot_out_t, dt, r"$\dot{M}_{\mathrm{out}}$", c[1])
    plot_band(ax[0, 1], r, Mdot_t, dt, r"$\dot{M}$", c[2])
    plot_band(ax[0, 1], rf, Mdotf_t, dt, r"$\dot{M}_f$", 'grey')
    plot_band(ax[0, 2], rf, Jdot_advf_t, dt, r"$\dot{J}_{\mathrm{adv}, f}$",
              c[0])
    plot_band(ax[0, 2], rf, Jdot_vf_t, dt, r"$\dot{J}_{\mathrm{visc}, f}$",
              c[1])
    plot_band(ax[0, 2], r, Jdot_gs_t, dt, r"$\dot{J}_{\mathrm{grav}, s}$",
              c[2])
    plot_band(ax[1, 0], r, Sig_t, dt, r"$\Sigma$", c[0],)
    plot_band(ax[1, 1], r, Mdot_in_t, dt, r"$\dot{M}_{\mathrm{in}}$", c[0])
    plot_band(ax[1, 1], r, Mdot_out_t, dt, r"$\dot{M}_{\mathrm{out}}$", c[1])
    plot_band(ax[1, 1], r, Mdot_t, dt, r"$\dot{M}$", c[2])
    plot_band(ax[1, 1], rf, Mdotf_t, dt, r"$\dot{M}_f$", 'grey')
    plot_band(ax[1, 2], rf, Jdot_advf_t, dt, r"$\dot{J}_{\mathrm{adv}, f}$",
              c[0])
    plot_band(ax[1, 2], rf, Jdot_vf_t, dt, r"$\dot{J}_{\mathrm{visc}, f}$",
              c[1])
    plot_band(ax[1, 2], r, Jdot_gs_t, dt, r"$\dot{J}_{\mathrm{grav}, s}$",
              c[2])

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[0, 2].legend()

    ax[0, 0].set(xlim=(rmin, rmax), ylabel=r'$\Sigma$')
    ax[0, 1].set(xlim=(rmin, rmax), ylabel=r'$\dot{M}$')
    ax[0, 2].set(xlim=(rmin, rmax), ylabel=r'$\dot{J}$')
    ax[1, 0].set(xlim=(rjph[1], rmax), xscale='log', yscale='log',
                 ylabel=r'$\Sigma$')
    ax[1, 1].set(xlim=(rjph[1], rmax), xscale='log', yscale='log',
                 ylabel=r'$\dot{M}$')
    ax[1, 2].set(xlim=(rjph[1], rmax), xscale='log', yscale='log',
                 ylabel=r'$\dot{J}$')

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

    
