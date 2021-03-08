"""equivalent to reading patho report"""

import argparse, os, os.path as op, numpy as np
import cutslib as cl
from cutslib.analysis import PathologyManager
from cutslib import util

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--odir", default='out')
parser.add_argument("--oname", default="report.txt")
parser.add_argument("--todlist")
parser.add_argument("--release", required=True)
parser.add_argument("--mpi", action='store_true')
parser.add_argument("--size", type=int, default=1)
parser.add_argument("--rank", type=int, default=0)
args = parser.parse_args()
# mpi setup
if args.mpi:
    from cutslib.mpi import COMM_WORLD
    comm = COMM_WORLD
    rank = comm.rank
    size = comm.size
else:  # assume fake parallel by default
    comm = None
    rank = args.rank
    size = args.size
# load release file and depot
rl = cl.Release(args.release)
todlist = cl.TODList.from_file(args.todlist)
depot = cl.Depot(rl.depot)
# initialize empty lists
n_lives = []; n_valids = []; tfracs  = []; n_gains = []; n_corrs = []; tnames = []
n_norms = []; n_rmss   = []; n_kurts = []; n_skews = []; n_resps = []; n_ffs  = []
for ti in range(rank, len(todlist), size):
    tname = todlist[ti]
    print(f'{ti+1:5d}/{len(todlist)}: {tname}')
    # load metadata only
    tod = cl.load_tod(tname, autoloads=[], verbose=False, rd=False)
    tag = rl.tod_tags(tname)['tag_out']
    # load pman
    try:
        pman = depot.read_object(PathologyManager, tag=tag, tod=tod)
    except: continue
    # quantity of interests
    # 1. live dets
    n_live = len(pman.cman.cuts['sel'].get_uncut())
    # 2. dets with valid calibrations (ff, resp)
    n_valid = np.sum((pman.patho['ff'][pman.live]!=0) * (pman.patho['resp'][pman.live]!=0))
    # 3. tfrac: partial cut fraction among live detector
    cuts  = pman.cman.cuts['sel']
    tfrac = 0
    for d in cuts.get_uncut():
        tfrac += float(np.sum(np.diff(cuts.cuts[d]))/tod.nsamps)
    if len(cuts.get_uncut()) > 0: tfrac /= len(cuts.get_uncut())
    # 4. individual crit live dets
    cman = pman.cman
    for k in cman.cuts:
        n_gain = n_corr = n_norm = n_rms = n_kurt = n_skew = n_resp = n_ff = 0
        if 'gain_sel' in cman.cuts: n_gain = len(cman.cuts['gain_sel'].get_uncut())
        if 'corr_sel' in cman.cuts: n_corr = len(cman.cuts['corr_sel'].get_uncut())
        if 'norm_sel' in cman.cuts: n_norm = len(cman.cuts['norm_sel'].get_uncut())
        if 'rms_sel'  in cman.cuts: n_rms  = len(cman.cuts['rms_sel' ].get_uncut())
        if 'kurt_sel' in cman.cuts: n_kurt = len(cman.cuts['kurt_sel'].get_uncut())
        if 'skew_sel' in cman.cuts: n_skew = len(cman.cuts['skew_sel'].get_uncut())
        if 'resp_sel' in cman.cuts: n_resp = len(cman.cuts['resp_sel'].get_uncut())
        if 'ff_sel'   in cman.cuts: n_ff   = len(cman.cuts['ff_sel'  ].get_uncut())
    # append results
    n_lives.append(n_live); n_valids.append(n_valid); tfracs.append(tfrac);
    n_gains.append(n_gain); n_corrs.append(n_corr); n_norms.append(n_norm);
    n_rmss.append(n_rms); n_kurts.append(n_kurt); n_skews.append(n_skew);
    n_resps.append(n_resp); n_ffs.append(n_ff); tnames.append(tname)
if not args.mpi:
    n_lives  = np.array(n_lives )
    n_valids = np.array(n_valids)
    tfracs   = np.array(tfracs  )
    n_gains  = np.array(n_gains )
    n_corrs  = np.array(n_corrs )
    n_norms  = np.array(n_norms )
    n_rmss   = np.array(n_rmss  )
    n_kurts  = np.array(n_kurts )
    n_skews  = np.array(n_skews )
    n_resps  = np.array(n_resps )
    n_ffs    = np.array(n_ffs   )
    tnames   = np.array(tnames  )
else:
    n_lives  = util.allgatherv(n_lives, comm)
    n_valids = util.allgatherv(n_valids,comm)
    tfracs   = util.allgatherv(tfracs,  comm)
    n_gains  = util.allgatherv(n_gains, comm)
    n_corrs  = util.allgatherv(n_corrs, comm)
    n_norms  = util.allgatherv(n_norms, comm)
    n_rmss   = util.allgatherv(n_rmss,  comm)
    n_kurts  = util.allgatherv(n_kurts, comm)
    n_skews  = util.allgatherv(n_skews, comm)
    n_resps  = util.allgatherv(n_resps, comm)
    n_ffs    = util.allgatherv(n_ffs,   comm)
    tnames   = util.allgatherv(tnames,  comm)
if rank == 0:
    ofile = op.join(args.odir, args.oname)
    print("Writing:", ofile)
    with open(ofile, "w") as f:
        f.write("#                          tod live  cal tfrac gain corr norm  rms kurt skew resp   ff\n")
        for i in range(len(tnames)):
            f.write(f"{tnames[i]} {n_lives[i]:4d} {n_valids[i]:4d} {tfracs[i]:.3f} {n_gains[i]:4d} "
                    f"{n_corrs[i]:4d} {n_norms[i]:4d} {n_rmss[i]:4d} {n_kurts[i]:4d} {n_skews[i]:4d} "
                    f"{n_resps[i]:4d} {n_ffs[i]:4d}\n")
