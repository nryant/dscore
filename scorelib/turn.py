"""Classes for representing speaker turns and interacting with RTTM files."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict

from intervaltree import IntervalTree

from .six import iterkeys, python_2_unicode_compatible
from .utils import clip, groupby, warn, xor

__all__ = ['merge_turns', 'trim_turns', 'Turn']


@python_2_unicode_compatible
class Turn(object):
    """Speaker turn class.

    A turn represents a segment of audio attributed to a single speaker.

    Parameters
    ----------
    onset : float
        Onset of turn in seconds from beginning of recording.

    offset : float, optional
        Offset of turn in seconds from beginning of recording. If None, then
        computed from ``onset`` and ``dur``.
        (Default: None)

    dur : float, optional
        Duration of turn in seconds. If None, then computed from ``onset`` and
        ``offset``.
        (Default: None)

    speaker_id : str, optional
        Speaker id.
        (Default: None)

    file_id : str, optional
        File id.
        (Default: none)
    """
    def __init__(self, onset, offset=None, dur=None, speaker_id=None,
                 file_id=None):
        if not xor(offset is None, dur is None):
            raise ValueError('Exactly one of offset or dur must be given')
        if onset < 0:
            raise ValueError('Turn onset must be >= 0 seconds')
        if offset:
            dur = offset - onset
        if dur <= 0:
            raise ValueError('Turn duration must be > 0 seconds')
        if not offset:
            offset = onset + dur
        self.onset = onset
        self.offset = offset
        self.dur = dur
        self.speaker_id = speaker_id
        self.file_id = file_id

    def __str__(self):
        return ('FILE: %s, SPEAKER: %s, ONSET: %f, OFFSET: %f, DUR: %f' %
                (self.file_id, self.speaker_id, self.onset, self.offset,
                 self.dur))

    def __repr__(self):
        speaker_id = ("'%s'" % self.speaker_id if self.speaker_id is not None
                      else None)
        file_id = ("'%s'" % self.file_id if self.file_id is not None
                   else None)
        return ('Turn(%f, %f, None, %s, %s)' %
                (self.onset, self.offset, speaker_id, file_id))


def merge_turns(turns):
    """Merge overlapping turns by same speaker within each file."""
    # Merge separately within each file and for each speaker.
    new_turns = []
    for (file_id, speaker_id), speaker_turns in groupby(
            turns, lambda x: (x.file_id, x.speaker_id)):
        speaker_turns = list(speaker_turns)
        speaker_it = IntervalTree.from_tuples(
            [(turn.onset, turn.offset) for turn in speaker_turns])
        n_turns_pre = len(speaker_it)
        speaker_it.merge_overlaps()
        n_turns_post = len(speaker_it)
        if n_turns_post < n_turns_pre:
            speaker_turns = []
            for intrvl in speaker_it:
                speaker_turns.append(
                    Turn(intrvl.begin, intrvl.end, speaker_id=speaker_id,
                         file_id=file_id))
            speaker_turns = sorted(
                speaker_turns, key=lambda x: (x.onset, x.offset))
            warn('Merging overlapping speaker turns. '
                 'FILE: %s, SPEAKER: %s' % (file_id, speaker_id))
        new_turns.extend(speaker_turns)
    turns = new_turns

    return turns


def trim_turns(turns, uem=None, score_onset=None, score_offset=None):
    """Trim turns to scoring regions defined in UEM.

    Parameters
    ----------
    turns : list of Turn
        Speaker turns.

    uem : UEM, optional
        Un-partitioned evaluation map.
        (Default: None)

    score_onset : float, optional
        Onset of scoring region in seconds from beginning of file. Only valid
        if ``uem=None``.
        (Default: None)

    score_offset : float, optional
        Offset of scoring region in seconds from beginning of file. Only
        valid if ``uem=None``.
        (Default: None)

    Returns
    -------
    trimmed_turns : list of Turn
        Trimmed turns.
    """
    if uem is not None:
        if not (score_onset is None and score_offset is None):
            raise ValueError('Either uem or score_onset and score_offset must '
                             'be specified.')
    else:
        if score_onset is None or score_offset is None:
            raise ValueError('Either uem or score_onset and score_offset must '
                             'be specified.')
        if score_onset < 0:
            raise ValueError('Scoring region onset must be >= 0 seconds')
        if score_offset <= score_onset:
            raise ValueError('Scoring region duration must be > 0 seconds')

    new_turns = []
    for turn in turns:
        if uem is not None:
            if turn.file_id not in uem:
                warn('Skipping turn from file not in UEM. TURN: %s' % turn)
                continue
            score_onset, score_offset = uem[turn.file_id]
        turn_onset = clip(turn.onset, score_onset, score_offset)
        turn_offset = clip(turn.offset, score_onset, score_offset)
        if turn.onset != turn_onset or turn.offset != turn_offset:
            warn('Truncating turn overlapping non-scoring region. TURN: %s ' %
                 turn)
        if turn_offset <= turn_onset:
            continue
        new_turns.append(Turn(
            turn_onset, turn_offset, speaker_id=turn.speaker_id,
            file_id=turn.file_id))
    return new_turns
