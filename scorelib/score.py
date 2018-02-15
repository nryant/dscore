"""Functions for scoring paired system/reference RTTM files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict

import numpy as np
from scipy.linalg import block_diag

from . import metrics
from .six import iteritems, itervalues, python_2_unicode_compatible

__all__ = ['score', 'turns_to_frames']


def turns_to_frames(turns, score_onset, score_offset, step=0.010,
                    as_string=False):
    """Return frame-level labels corresponding to diarization.

    Parameters
    ----------
    turns : list of Turn
        Speaker turns. Should all be from single file.

    score_onset : float
        Scoring region onset in seconds from beginning of file.

    score_offset : float
        Scoring region offset in seconds from beginning of file.

    step : float, optional
        Frame step size  in seconds.
        (Default: 0.01)

    as_string : bool, optional
        If True, returned frame labels will be strings that are the class
        names. Else, they will be integers.

    Returns
    -------
    labels : ndarray, (n_frames,)
        Frame-level labels.
    """
    file_ids = set([turn.file_id for turn in turns])
    if score_offset <= score_onset:
        raise ValueError('score_onset must be less than score_offset: '
                         '%.3f >= %.3f' % (score_onset, score_offset))
    if len(file_ids) > 1:
        raise ValueError('Turns should be from a single file.')

    # Create matrix whose i,j-th entry is True IFF the j-th speaker was
    # present at frame i.
    onsets = [turn.onset for turn in turns]
    offsets = [turn.offset for turn in turns]
    speaker_ids = [turn.speaker_id for turn in turns]
    speaker_classes, speaker_class_inds = np.unique(
        speaker_ids, return_inverse=True)
    speaker_classes = np.concatenate([speaker_classes, ['non-speech']])
    dur = score_offset - score_onset
    n_frames = int(dur/step)
    X = np.zeros((n_frames, speaker_classes.size), dtype='bool')
    times = score_onset + step*np.arange(n_frames)
    bis = np.searchsorted(times, onsets)
    eis = np.searchsorted(times, offsets)
    for bi, ei, speaker_class_ind in zip(bis, eis, speaker_class_inds):
        X[bi:ei, speaker_class_ind] = True
    is_nil = ~(X.any(axis=1))
    X[is_nil, -1] = True

    # Now, convert to frame-level labelings.
    pows = 2**np.arange(X.shape[1])
    labels = np.sum(pows*X, axis=1)
    if as_string:
        def speaker_mask(n):
            return [bool(int(x))
                    for x in np.binary_repr(n, speaker_classes.size)][::-1]
        label_classes = np.array(['_'.join(speaker_classes[speaker_mask(n)])
                                  for n in range(2**speaker_classes.size)])
        try:
            # Save some memory in the (majority of) cases where speaker ids are
            # ASCII.
            label_classes = label_classes.astype('string')
        except UnicodeEncodeError:
            pass
        labels = label_classes[labels]
    return labels


@python_2_unicode_compatible
class Scores(object):
    """Structure containing metrics.

    Parameters
    ----------
    der : float
        Diarization error rate in percent.

    bcubed_precision : float
        B-cubed precision.

    bcubed_recall : float
        B-cubed recall.

    bcubed_f1 : float
        B-cubed F1.

    tau_ref_sys : float
        Value between 0 and 1 that is high when the reference diarization is
        predictive of the system diarization and low when the reference
        diarization provides essentially no information about the system
        diarization.

    tau_sys_ref : float
        Value between 0 and 1 that is high when the system diarization is
        predictive of the reference diarization and low when the system
        diarization provides essentially no information about the reference
        diarization.

    ce_ref_sys : float
        Conditional entropy of the reference diarization given the system
        diarization.

    ce_sys_ref : float
        Conditional entropy of the system diarization given the reference
        diarization.

    mi : float
        Mutual information.

    nmi : float
        Normalized mutual information.
    """
    def __init__(self, der, bcubed_precision, bcubed_recall, bcubed_f1,
                 tau_ref_sys, tau_sys_ref, ce_ref_sys, ce_sys_ref, mi, nmi):
        self.der = der
        self.bcubed_precision = bcubed_precision
        self.bcubed_recall = bcubed_recall
        self.bcubed_f1 = bcubed_f1
        self.tau_ref_sys = tau_ref_sys
        self.tau_sys_ref = tau_sys_ref
        self.ce_ref_sys = ce_ref_sys
        self.ce_sys_ref = ce_sys_ref
        self.mi = mi
        self.nmi = nmi

    def __str__(self):
        return ('DER: %.2f, B-cubed precision: %.2f, B-cubed recall: %.2f, '
                'B-cubed F1: %.2f, GKT(ref, sys): %.2f, GKT(sys, ref): %.2f, '
                'CE(ref|sys): %.2f, CE(sys|ref): %.2f, MI: %.2f, NMI: %.2f' %
                (self.der, self.bcubed_precision, self.bcubed_recall,
                 self.bcubed_f1, self.tau_ref_sys, self.tau_sys_ref,
                 self.ce_ref_sys, self.ce_sys_ref, self.mi, self.nmi))


def score(ref_turns, sys_turns, uem, der_collar=0.0,
          der_ignore_overlaps=True, step=0.010, nats=False):
    """Score diarization.

    Parameters
    ----------
    ref_turns : list of Turn
        Reference speaker turns.

    sys_turns : list of Turn
        System speaker turns.

    uem : UEM
        Un-partitioned evaluation map.

    der_collar : float, optional
        Size of forgiveness collar in seconds to use in computing Diarization
        Erro Rate (DER). Diarization output will not be evaluated within +/-
        ``collar`` seconds of reference speaker boundaries.
        (Default: 0.0)

    der_ignore_overlaps : bool, optional
        If True, ignore regions in the reference diarization in which more
        than one speaker is speaking when computing DER.
        (Default: True)

    step : float, optional
        Frame step size  in seconds. Not relevant for computation of DER.
        (Default: 0.01)

    nats : bool, optional
        If True, use nats as unit for information theoretic metrics.
        Otherwise, use bits.
        (Default: False)

    Returns
    -------
    file_to_scores : dict
        Mapping from file ids in ``uem`` to ``Scores`` instances.

    global_scores : Scores
        Global scores.
    """
    def groupby(turns):
        file_to_turns = defaultdict(list)
        for turn in turns:
            file_to_turns[turn.file_id].append(turn)
        return file_to_turns
    file_to_ref_turns = groupby(ref_turns)
    file_to_sys_turns = groupby(sys_turns)

    # Build contingency matrices.
    file_to_cm = {}
    for file_id, (score_onset, score_offset) in iteritems(uem):
        ref_labels = turns_to_frames(
            file_to_ref_turns[file_id], score_onset, score_offset, step=step)
        sys_labels = turns_to_frames(
            file_to_sys_turns[file_id], score_onset, score_offset, step=step)
        file_to_cm[file_id], _, _ = metrics.contingency_matrix(
            ref_labels, sys_labels)
    global_cm = block_diag(*list(itervalues(file_to_cm)))

    # Score.
    def compute_metrics(cm):
        bcubed_precision, bcubed_recall, bcubed_f1 = metrics.bcubed(
            None, None, cm)
        tau_ref_sys, tau_sys_ref = metrics.goodman_kruskal_tau(
            None, None, cm)
        ce_ref_sys = metrics.conditional_entropy(None, None, cm, nats)
        ce_sys_ref = metrics.conditional_entropy(None, None, cm.T, nats)
        mi, nmi = metrics.mutual_information(None, None, cm, nats)
        return Scores(None, bcubed_precision, bcubed_recall, bcubed_f1,
                      tau_ref_sys, tau_sys_ref, ce_ref_sys, ce_sys_ref,
                      mi, nmi)
    file_to_der, global_der = metrics.der(
        ref_turns, sys_turns, der_collar, der_ignore_overlaps, uem)
    file_to_scores = {}
    for file_id, cm in iteritems(file_to_cm):
        scores = compute_metrics(cm)
        scores.der = file_to_der[file_id]
        file_to_scores[file_id] = scores
    global_scores = compute_metrics(global_cm)
    global_scores.der = global_der

    return file_to_scores, global_scores
