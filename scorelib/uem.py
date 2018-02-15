"""Functions for reading/writing and manipulating NIST un-partitioned
evaluation maps.

An un-partitioned evaluation map (UEM) specifies the time regions within each
file that will be scored.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict, Iterable, Mapping
import itertools
import os

from .six import iteritems, iterkeys
from .utils import format_float

__all__ = ['gen_uem', 'load_uem', 'write_uem', 'UEM']


class UEM(dict):
    """Un-partitioned evaluaion map (UEM).

    A UEM defines a mapping from file ids to scoring regions.
    """
    def __setitem__(self, k, v):
        if not isinstance(v, Iterable):
            raise ValueError('Not a valid interval.')
        v = list(v)
        if len(v) != 2:
            raise ValueError('Not a valid interval.')
        onset, offset = v
        onset = float(onset)
        offset = float(offset)
        if onset >= offset:
            raise ValueError('Not a valid interval.')
        super(UEM, self).__setitem__(k, (onset, offset))

    def update(self, iterable, **kwargs):
        if iterable is not None:
            if isinstance(iterable, Mapping):
                for k, v in iteritems(iterable):
                    self.__setitem__(k, v)
            else:
                for k, v in iterable:
                    self.__setitem__(k, v)
        if kwargs:
            for k, v in iteritems(kwargs):
                self.__setitem__(k, v)


def load_uem(uemf):
    """Load un-partitioned evaluation map from file in NIST format.

    The un-partitioned evaluation map (UEM) file format contains
    one record per line, each line consisting of NN space-delimited
    fields:

    - file id  --  file id
    - channel  --  channel (1-indexed)
    - onset  --  onset of evaluation region in seconds from beginning of file
    - offset  --  offset of evaluation region in seconds from beginning of
      file

    Lines beginning with semicolons are regarded as comments and ignored.

    Parameters
    ----------
    uemf : str
        Path to UEM file.

    Returns
    -------
    uem : UEM
        Evaluation map.
    """
    with open(uemf, 'rb') as f:
        uem = UEM()
        for line in f:
            if line.startswith(b';'):
                continue
            fields = line.decode('utf-8').strip().split()
            file_id = os.path.splitext(fields[0])[0]
            onset = float(fields[2])
            offset = float(fields[3])
            uem[file_id] = (onset, offset)
    return uem


def write_uem(uemf, uem, n_digits=3):
    """Write un-partitioned evaluation map to file in NIST format.

    Parameters
    ----------
    uemf : str
        Path to output UEM file.

    uem : UEM
        Evaluation map.

    n_digits : int, optional
        Number of decimal digits to round to.
        (Default: 3)
    """
    with open(uemf, 'wb') as f:
        for file_id in sorted(iterkeys(uem)):
            onset, offset = uem[file_id]
            line = ' '.join([file_id,
                             '1',
                             format_float(onset, n_digits),
                             format_float(offset, n_digits)
                            ])
            f.write(line.encode('utf-8'))
            f.write(b'\n')


def gen_uem(ref_turns, sys_turns):
    """Generate un-partitioned evaluation map.

    For each file, the extent of the scoring region is set as follows:

    - onset = min(minimum reference onset, minimum system onset)
    - offset = max(maximum reference onset, maximum system offset)

    Parameters
    ----------
    ref_turns : list of Turn
        Reference speaker turns.

    sys_turns : list of Turn
        System speaker turns.

    Returns
    -------
    uem : UEM
        Un-partitioned evaluation map.
    """
    file_ids = set()
    onsets = defaultdict(set)
    offsets = defaultdict(set)
    for turn in itertools.chain(ref_turns, sys_turns):
        file_ids.add(turn.file_id)
        onsets[turn.file_id].add(turn.onset)
        offsets[turn.file_id].add(turn.offset)
    uem = UEM()
    for file_id in file_ids:
        uem[file_id] = (min(onsets[file_id]), max(offsets[file_id]))
    return uem
