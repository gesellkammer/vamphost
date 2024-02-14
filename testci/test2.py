from __future__ import annotations
import numpy as np
import vamp
import argparse
import sndfileio
import sys


def pyinPitchTrack(samples: np.ndarray,
                   sr: int,
                   fftSize=2048,
                   overlap=8,
                   lowAmpSuppression=0.1,
                   threshDistr="beta15",
                   onsetSensitivity=0.7,
                   pruneThresh=0.1,
                   outputUnvoiced='negative'
                   ) -> np.ndarray:
    """
    Analyze the samples and extract fundamental and voicedness

    For each measurement calculates the fundamental frequency and the voicedness
    probability  (the confidence that sound is pitched at a given time).

    pYIN vamp plugin: https://code.soundsoftware.ac.uk/projects/pyin/files

    Args:
        samples: the audio samples (mono). If a multichannel sample
            is given, only the first channel will be processed
        sr (int): sample rate
        fftSize: fft size (vamp names this "block_size"). Must be >= 2048
        overlap: determines the hop size (hop size = fftSize // overlap)
        lowAmpSuppression: supress low amplitude pitch estimates
        threshDistr: yin threshold distribution. See table 1 below
        onsetSensitivity: onset sensitivity
        pruneThresh: totalDuration pruning threshold
        outputUnvoiced: method used to output frequencies when the sound is
            unvoiced (there is no reliable pitch detected). Choices are True (sets
            the frequency to 'nan' for unvoiced breakpoints), False (the breakpoint
            is skipped) or 'negative' (outputs the detected frequency as negative)


    Returns:
        a 2D numpy array (float32) of 3 column with one row for each step.
        The columns are: time, f0, voiced probability (~= confidence)

        Whenever the confidence that the f0 is correct drops below
        a certain threshold, the frequency is given as negative

    ============   ============
    thresh_distr   Description
    ============   ============
    uniform        Uniform
    beta10         Beta (mean 0.10)
    beta15         Beta (mean 0.15)
    beta30         Beta (mean 0.30)
    single10       Single value 0.10
    single15       Single value 0.15
    single20       Single value 0.20
    ============   ============

    Example::

        import sndfileio
        import csoundengine as ce
        from from maelzel.snd.vamptools import *
        samples, sr = sndfileio.sndread("/path/to/soundfile.wav")
        matrix = pyin_pitchtrack(samples, sr)
        times = matrix[:,0]
        freqs = matrix[:,1]
        TODO
        freqbpf = bpf.core.Linear(times, matrix[:,1])
        midibpf = freqbpf.f2m()[::0.05]
        voicedprob = bpf.core.Linear(times, matrix[:,2])
        # play both the sample and the f0 to check
        tabnum = ce.makeTable(samples, sr=sr, block=True)
        ce.playSample(tabnum)
        synth = ce.session().sched('.sine', pargs={'kmidi':pitch(0)})
        synth.automatePargs('kmidi', pitch.flat_pairs())
        synth.automatePargs('kamp', pitchpitch.flat_pairs())

        TODO
    """
    allplugins = vamp.list_plugins()
    if 'pyin:pyin' not in allplugins:
        raise RuntimeError(f"Vamp plugin 'pyin' not found. Available plugins: {allplugins}")

    if fftSize < 2048:
        raise ValueError("The pyin vamp plugin does not accept fft size less than 2048")

    if len(samples.shape) > 1:
        samples = samples[:,0]
    threshdistridx = _pyinThresholdDistrs.get(threshDistr)
    if threshdistridx is None:
        raise ValueError(f"Unknown threshold distribution: {threshDistr}. "
                         f"It must be one of {', '.join(_pyinThresholdDistrs.keys())}")

    output_unvoiced_idx = {
        False: 0,
        True: 1,
        "negative": 2,
        "nan": 2
    }.get(outputUnvoiced)

    if output_unvoiced_idx is None:
        raise ValueError(f"Unknown output_unvoiced value {outputUnvoiced}. "
                         f"possible values: {False, True, 'negative'}")
    step_size = fftSize // overlap
    kwargs = {'step_size': step_size, "block_size": fftSize}
    plugin_key = "pyin:pyin"
    params = {
        'lowampsuppression': lowAmpSuppression,
        'onsetsensitivity': onsetSensitivity,
        'prunethresh': pruneThresh,
        'threshdistr': threshdistridx,
        'outputunvoiced': output_unvoiced_idx
    }
    plugin, step_size, block_size = vamp.load.load_and_configure(samples, sr, plugin_key,
                                                                 params, **kwargs)

    ff = vamp.frames.frames_from_array(samples, step_size, block_size)
    outputs = ['smoothedpitchtrack', 'voicedprob', 'f0candidates']
    results = list(vamp.process.process_with_initialised_plugin(ff,
                                                                sample_rate=sr,
                                                                step_size=step_size,
                                                                plugin=plugin,
                                                                outputs=outputs))

    vps = [d['voicedprob'] for d in results if 'voicedprob' in d]
    pts = [d['smoothedpitchtrack'] for d in results if 'smoothedpitchtrack' in d]
    f0s = [d['f0candidates'] for d in results if 'f0candidates' in d]

    arr = np.empty((len(vps), 3))
    i = 0
    NAN = float('nan')
    for vp, pt, f0 in zip(vps, pts, f0s):
        t = vp['timestamp']
        probs = vp['values']
        candidates = f0.get('values', None)
        freq = float(pt['values'][0])
        if freq < 0:
            if outputUnvoiced == 'nan':
                freq = NAN
            prob = 0
        elif candidates is None:
            prob = 0
        else:
            candidates = candidates.astype('float64')
            if len(candidates) == len(probs):
                idx = numpyx.nearestidx(candidates, freq, sorted=False)
                prob = probs[idx]
            else:
                prob = probs[0]
        arr[i] = [t, freq, prob]
        i += 1
    return arr


# -------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('sndfile')
parser.add_argument('--mode', default='pitchtrack')
args = parser.parse_args()

samples, sr = sndfileio.sndread(args.sndfile)

if args.mode == 'pitchtrack':
    outarr = pyinPitchTrack(samples=samples, sr=sr)
    print(outarr)

else:
    print(f"ERROR: Unknown mode: {args.mode}")
    sys.exit(-1)


