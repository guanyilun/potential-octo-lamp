"""This script aims to be a short standalone script that derive
pathology and generates cuts and cal for a list of TODs.

Note: 
- source cuts, planet cuts, find jump were not included
- MFE, DE are not calculated
- It assumes that no dark detector is available.
- partial cuts not written to disk by default
- assume fake parallel by default unless specified with --mpi
- it (mis)uses release file and uses it to find the relevant
  parameter files
- tags in cutparam will be ignored
- plan to change parameter files into a filedb like syntax in
  the future
"""

import argparse, os, os.path as op
import numpy as np
from enlib import bench
import moby2

import cutslib as cl
from cutslib.release import Release
from cutslib import analysis as ana, preselect, load, errors
from cutslib.analysis import CutsManager, PathologyManager
from cutslib.util import dets2sel

parser = argparse.ArgumentParser()
parser.add_argument('--todlist', required=True)
parser.add_argument('--release', help='release filepath', required=True)
parser.add_argument('--mpi', action='store_true')
parser.add_argument('--size', help='fmpi size', type=int, default=1)
parser.add_argument('--rank', help='fmpi rank', type=int, default=0)
parser.add_argument('--limit', help='debug: run only a few tods', type=int, default=None)
# parser.add_argument('-o', '--odir', default='out')
# parser.add_argument('-v', action='store_true')
args = parser.parse_args()
# if not op.exists(args.odir): os.makedirs(args.odir)

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
    
# load release file which specifies where params are to be loaded
rl = Release(args.release)
# load depot for I/O
depot = cl.Depot(rl.depot)
# loop over tods
todlist = cl.TODList.from_file(args.todlist)
# optionally limit todlist for debugging
if args.limit: todlist = todlist[:args.limit]
for ti in range(rank, len(todlist), size):
    tname = todlist[ti]
    print(f'{ti+1:5d}/{len(todlist)}: {tname}')
    # load tod
    with bench.show("load tod"):
        tod = cl.load_tod(tname, autoloads=[], verbose=False)
    # load cut params
    cutparam = rl.tod_cutparam(tname)
    cutParam = rl.tod_cutParam(tname)
    pathop   = cutParam.get('pathologyParams')
    # load tags
    tags     = rl.tod_tags(tname)
    tag_out  = tags['tag_out']
    if not 'tag_partial' in tags:
        tag_partial = f"{tag_out}_partial"
    else:
        tag_partial = tags['tag_partial'] 
    # source cuts and planet cuts to be added here
    # ... 
    # simple preprocessing
    with bench.show('remove mean'):
        moby2.tod.remove_mean(tod)
    with bench.show('remove filter gain'):        
        moby2.tod.remove_filter_gain(tod)
    # initialize manager for cuts and pathologies
    cman = CutsManager.for_tod(tod)
    pman = PathologyManager.for_tod(tod)
    # find mce cuts and glitch cuts
    # if partial cuts already exist, skip otherwise proceed
    # by default we don't write partial cuts
    with bench.show('cut partial'):
        cuts = load.get_partial(tod, cutParam.get('glitchParams'),
                                tag=tag_partial, depot=depot,
                                force=False,  # FIXME: for debugging
                                write=True)
        # store in manager
        cman.add('glitch', cuts)
        # fill glitches before proceeding
        moby2.tod.fill_cuts(tod, cuts, extrapolate=False, no_noise=True)
    # downsample before proceeding
    with bench.show('downsampling'):
        tod = tod.copy(resample=2, resample_offset=1)
    # find detectors that are all zeros
    zero_sel = tod.data[:,::100].any(axis=1)
    cman.add('zero_sel', zero_sel)
    # remove detector that fluctuates over the full
    # dynamic range
    range_sel = np.std(tod.data,axis=1) < 1e8
    cman.add('range_sel', range_sel)
    # analyze scan
    with bench.show('analyze scan'):
        scan_params = ana.analyze_scan(tod)
        for f_ in ['scan_flags', 'turn_flags']:
            cman.add(f_, cl.CutsVector.from_mask(scan_params[f_]))
    ###############
    # calibration #
    ###############
    # get flatfield
    ff, ff_sel, stable = load.get_ff(tod, pathop['calibration']['flatfield'])
    # get responsivity from biasstep or IV
    resp, resp_sel = load.get_resp(tod, pathop['calibration']['config'])
    # save various cuts
    cman.add('resp_sel', resp_sel)
    cman.add('ff_sel', ff_sel)
    cman.add('ff_stable', stable)
    # also save the pathologies
    pman.add('resp', resp)
    pman.add('ff', ff)
    # get detector lists
    live, dark, excl = load.get_dets(pathop['detectorLists'])
    # calibrate tod
    with bench.show("calibration"):
        cal = resp * ff
        moby2.tod.apply_calibration(tod.data, tod.det_uid, cal)
    # find jump skipped
    #################
    # fft transform #
    #################
    with bench.show("detrend"):
        trend = moby2.tod.detrend_tod(tod)
    with bench.show("fft"):
        fdata = moby2.tod.fft(tod.data)
        freqs = moby2.tod.filters.genFreqsTOD(tod)
    ############################
    # begin multifreq analysis #
    ############################
    # low-freq analysis
    lf_param = {'fmin': 0.02, 'fmax': 0.1, 'min_corr': 0.9}  # FIXME: temp params
    fmask = (freqs > lf_param['fmin']) * (freqs < lf_param['fmax'])
    dets = dets2sel(live, len(tod.det_uid)) * (cal != 0) * zero_sel * range_sel
    preselector = preselect.by_median(min_corr=lf_param['min_corr'], dets=dets)
    with bench.show("analyze common mode"):
        try:
            cm = ana.analyze_common_mode(fdata[:,fmask], nsamps=tod.nsamps,
                                         preselector=preselector, pman=pman, det_uid=dets)
        except errors.PreselectionError:
            print("Preselection failed, skipping")
            continue
    # high-freq analysis
    hf_param = {'fmin': 10, 'fmax': 20, 'n_deproject': 10}  # FIXME: temp params
    fmask = (freqs > hf_param['fmin']) * (freqs < hf_param['fmax'])
    with bench.show("analyze detector noise"):
        nm = ana.analyze_detector_noise(fdata[:,fmask], pman=pman, nsamps=tod.nsamps,
                                        n_deproject=hf_param['n_deproject'])
    ################
    # save results #
    ################
    # prepare cuts
    spar = pathop['liveSelParams']
    # redo calibration to get back to reasonable numbers
    recal = pman.patho['ff']*pman.patho['resp']
    recal[recal==0] = 1  # avoid warnings, zero anyway
    pman.patho['rms']  /= recal
    pman.patho['norm'] /= recal
    # this makes sure reletive limits are refering to
    # the dets only
    pman.restrict_dets(dets)
    for f, par in spar.items():
        if not par['apply']: continue
        pman.add_crit(field=f, limits=par['absCrit'], method='abs')
    cuts = pman.apply_crit(cman).combine_cuts(target='sel')
    # prepare calibration
    cal = pman.get_calibration(dets=live)
    # prepare patho
    # narrow pathologies to the live candidates only
    # save pathologies
    # bind useful info such as cuts manager, det lists into patho manager
    pman.cman = cman
    pman.live = live
    pman.excl = excl
    pman.dark = dark
    # drop the correlation matrices to save disk space
    pman.drop('cc').drop('c')
    # write results to disk
    # add exception handling to avoid unlikely racing condition
    depot.write_object(cuts, tod=tod, tag=tag_out)
    depot.write_object(cal,  tod=tod, tag=tag_out)
    depot.write_object(pman, tod=tod, tag=tag_out)

        
