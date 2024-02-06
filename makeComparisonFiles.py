from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import discopy.util as util
import discopy.geom as geom


def getDiagnostics(filename):

    pars = util.loadPars(filename)
    opts = util.loadOpts(filename)

    t, r, phi, z, prim, dat = util.loadCheckpoint(filename)
    _, rf, zf, f, fv, dt, rjph, zkph = util.loadFluxR(filename)
    _, rs, zs, s, sg, sv, ss, sc, sd, _, _, _ = util.loadSource(filename)
    _, _, _, diag, _, _ = util.loadDiagRZ(filename)

    Nr = rjph.shape[0] - 3

    rf = rf[0, :Nr]
    rs = rs[0, :Nr]
    rjph = rjph[:Nr+1]
    f = f[0, :Nr, :]
    fv = fv[0, :Nr, :]
    s = s[0, :Nr, :]
    sg = sg[0, :Nr, :]
    sv = sv[0, :Nr, :]
    ss = ss[0, :Nr, :]
    diag = diag[0, :Nr, :]

    return dict(t=t, rjph=rjph, rf=rf, rs=rs, f=f, fv=fv, s=s, sg=sg, sv=sv,
                ss=ss, diag=diag)


def extractComparisonVals(diagDict):

    t = diagDict['t']

    rjph = diagDict['rjph']
    dr = rjph[1:] - rjph[:-1]
    r = diagDict['rs']
    sg = diagDict['sg']
    ss = diagDict['ss']
    diag = diagDict['diag']

    Tg = -sg[:, 3].sum()

    j1 = np.searchsorted(rjph, 1.0) - 1
    j6 = np.searchsorted(rjph, 6.0) - 1

    Tg1 = -sg[j1:, 3].sum()

    dV = 2*np.pi* (0.5*(rjph[1:]+rjph[:-1])) * dr

    PsiR = (r*diag[:, 7]*dV).sum()
    PsiI = (r*diag[:, 8]*dV).sum()

    Sig = diag[:, 0]*dV
    SigEx = diag[:, 2]*dV
    SigEy = diag[:, 3]*dV

    ex = SigEx[j1:j6].sum() / Sig[j1:j6].sum()
    ey = SigEy[j1:j6].sum() / Sig[j1:j6].sum()

    Mdot = -ss[:, 0].sum()

    return t/(2*np.pi), np.array([Tg, Tg1, PsiR, PsiI, ex, ey, Mdot, 0])


def makeDiagnosticsFile(checkpointFiles, outFileName):

    N = len(checkpointFiles)

    t = np.empty(N)
    vals = np.empty((N, 8))

    for i, filename in enumerate(checkpointFiles):

        diag = getDiagnostics(filename)
        t[i], vals[i, :] = extractComparisonVals(diag)

    t_avg = 0.5*(t[1:] + t[:-1])

    vals = vals[1:, :]

    print("Saving", outFileName)
    with open(outFileName, 'w') as f:
        for i in range(N-1):
            strVals = ['{0:.15e}'.format(x) for x in vals[i]]
            f.write('{0:.15e} {1:s}\n'.format(t_avg[i], " ".join(strVals)))


def makeTimeseries(reportFile, pars, label):

    GM = 1.0
    a = 1.0

    omB = np.sqrt(GM/a**3)
    T = 2*np.pi * np.sqrt(a**3/GM)

    Sig0 = pars['Init_Par1']
    M0 = Sig0 * a**2
    Jdot0 = M0 * a**2 * omB**2
    Mdot0 = M0 * omB

    rep = util.CBDiscoReport(reportFile)

    t = rep.t[1:-1] / T

    dt = rep.t[2:] - rep.t[:-2]

    dM = rep.dM_Pl
    Mdot = (dM[:, 1:-1]+dM[:, 2:]) / dt[None, :]
    Mdot1 = Mdot[0] / Mdot0
    Mdot2 = Mdot[1] / Mdot0

    dJ_grv = rep.dJgrv_Pl.sum(axis=0) / Jdot0
    Jdot_grv = (dJ_grv[1:-1] + dJ_grv[2:]) / dt

    Jdot_grv_exc = -rep.Jdot_grv_exc_Gas.sum(axis=0)[1:-1] / Jdot0

    PsiR = rep.Psi1R[1:-1] / M0
    PsiI = rep.Psi1I[1:-1] / M0

    ex = rep.dMex_cav.sum(axis=0)[1:-1] / M0
    ey = rep.dMey_cav.sum(axis=0)[1:-1] / M0

    out_filename = "{0:s}.dat".format(label)

    N = len(t)

    vals = np.array([t, Jdot_grv, Jdot_grv_exc, PsiR, PsiI,
                     ex, ey, Mdot1, Mdot2])

    print("Writing", out_filename)
    with open(out_filename, "w") as f:
        for i in range(N):
            strvals = ["{0:.15e}".format(x) for x in vals[:, i]]
            line = " ".join(strvals)
            f.write(line + "\n")

    print("Done.")


def makeSnapshotBin(chkFile, label):

    t, r, phi, z, prim, dat = util.loadCheckpoint(chkFile)


    cos = np.cos(phi)
    sin = np.sin(phi)

    x = r * cos
    y = r * sin
    rho = prim[:, 0]
    vr = prim[:, 2]
    vp = r*prim[:, 3]

    vx = cos*vr - sin*vp
    vy = sin*vr + cos*vp

    N = r.shape[0]

    buf = np.empty((N, 5), dtype=float)
    buf[:, 0] = x
    buf[:, 1] = y
    buf[:, 2] = rho
    buf[:, 3] = vx
    buf[:, 4] = vy

    norb = int(t / (2*np.pi) + 0.5)

    filename = "{0:s}_{1:04d}.bin".format(label, norb)

    print("Writing", filename)

    buf.tofile(filename)

    print("Done.")


if __name__ == "__main__":

    filenames = [Path(x) for x in sys.argv[1:-1]]
    reportFile = filenames[0]
    checkpointFiles = filenames[1:]
    name = sys.argv[-1]

    pars = util.loadPars(checkpointFiles[0])

    label = "ryan_disco_{0:s}_{1:04d}".format(name, pars['Num_R'])

    makeTimeseries(reportFile, pars, label)
    
    for filename in checkpointFiles:
        makeSnapshotBin(filename, label)
