# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
#          simplified BSD-3 license

import datetime
import time
import os
import os.path as op


import numpy as np
import xml.etree.ElementTree as ElementTree


from ..base import BaseRaw, _check_update_montage
from ..utils import _read_segments_file, _create_chs
from ..meas_info import _empty_info
from ..constants import FIFF
from ...utils import verbose, logger, warn


def _read_header(fid):
    """Read EGI binary header."""
    version = np.fromfile(fid, np.int32, 1)[0]

    if version > 6 & ~np.bitwise_and(version, 6):
        version = version.byteswap().astype(np.uint32)
    else:
        ValueError('Watchout. This does not seem to be a simple '
                   'binary EGI file.')

    def my_fread(*x, **y):
        return np.fromfile(*x, **y)[0]

    info = dict(
        version=version,
        year=my_fread(fid, '>i2', 1),
        month=my_fread(fid, '>i2', 1),
        day=my_fread(fid, '>i2', 1),
        hour=my_fread(fid, '>i2', 1),
        minute=my_fread(fid, '>i2', 1),
        second=my_fread(fid, '>i2', 1),
        millisecond=my_fread(fid, '>i4', 1),
        samp_rate=my_fread(fid, '>i2', 1),
        n_channels=my_fread(fid, '>i2', 1),
        gain=my_fread(fid, '>i2', 1),
        bits=my_fread(fid, '>i2', 1),
        value_range=my_fread(fid, '>i2', 1)
    )

    unsegmented = 1 if np.bitwise_and(version, 1) == 0 else 0
    precision = np.bitwise_and(version, 6)
    if precision == 0:
        RuntimeError('Floating point precision is undefined.')

    if unsegmented:
        info.update(dict(n_categories=0,
                         n_segments=1,
                         n_samples=np.fromfile(fid, '>i4', 1)[0],
                         n_events=np.fromfile(fid, '>i2', 1)[0],
                         event_codes=[],
                         category_names=[],
                         category_lengths=[],
                         pre_baseline=0))
        for event in range(info['n_events']):
            event_codes = ''.join(np.fromfile(fid, 'S1', 4).astype('U1'))
            info['event_codes'].append(event_codes)
        info['event_codes'] = np.array(info['event_codes'])
    else:
        raise NotImplementedError('Only continuous files are supported')
    info['unsegmented'] = unsegmented
    info['dtype'], info['orig_format'] = {2: ('>i2', 'short'),
                                          4: ('>f4', 'float'),
                                          6: ('>f8', 'double')}[precision]
    info['dtype'] = np.dtype(info['dtype'])
    return info


def _read_events(fid, info):
    """Read events."""
    events = np.zeros([info['n_events'],
                       info['n_segments'] * info['n_samples']])
    fid.seek(36 + info['n_events'] * 4, 0)  # skip header
    for si in range(info['n_samples']):
        # skip data channels
        fid.seek(info['n_channels'] * info['dtype'].itemsize, 1)
        # read event channels
        events[:, si] = np.fromfile(fid, info['dtype'], info['n_events'])
    return events


def _combine_triggers(data, remapping=None):
    """Combine binary triggers."""
    new_trigger = np.zeros(data.shape[1])
    if data.astype(bool).sum(axis=0).max() > 1:  # ensure no overlaps
        logger.info('    Found multiple events at the same time '
                    'sample. Cannot create trigger channel.')
        return
    if remapping is None:
        remapping = np.arange(data) + 1
    for d, event_id in zip(data, remapping):
        idx = d.nonzero()
        if np.any(idx):
            new_trigger[idx] += event_id
    return new_trigger


@verbose
def read_raw_egi(input_fname, montage=None, eog=None, misc=None,
                 include=None, exclude=None, preload=False, verbose=None):
    """Read EGI simple binary as raw object.

    .. note:: The trigger channel names are based on the
              arbitrary user dependent event codes used. However this
              function will attempt to generate a synthetic trigger channel
              named ``STI 014`` in accordance with the general
              Neuromag / MNE naming pattern.

              The event_id assignment equals
              ``np.arange(n_events - n_excluded) + 1``. The resulting
              `event_id` mapping is stored as attribute to the resulting
              raw object but will be ignored when saving to a fiff.
              Note. The trigger channel is artificially constructed based
              on timestamps received by the Netstation. As a consequence,
              triggers have only short durations.

              This step will fail if events are not mutually exclusive.

    Parameters
    ----------
    input_fname : str
        Path to the raw file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Default is None.
    include : None | list
       The event channels to be ignored when creating the synthetic
       trigger. Defaults to None.
       Note. Overrides `exclude` parameter.
    exclude : None | list
       The event channels to be ignored when creating the synthetic
       trigger. Defaults to None. If None, channels that have more than
       one event and the ``sync`` and ``TREV`` channels will be
       ignored.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).

        ..versionadded:: 0.11

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : Instance of RawEGI
        A Raw object containing EGI data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawEGI(input_fname, montage, eog, misc, include, exclude, preload,
                  verbose)


class RawEGI(BaseRaw):
    """Raw object from EGI simple binary file."""

    @verbose
    def __init__(self, input_fname, montage=None, eog=None, misc=None,
                 include=None, exclude=None, preload=False,
                 verbose=None):  # noqa: D102
        if eog is None:
            eog = []
        if misc is None:
            misc = []
        with open(input_fname, 'rb') as fid:  # 'rb' important for py3k
            logger.info('Reading EGI header from %s...' % input_fname)
            egi_info = _read_header(fid)
            logger.info('    Reading events ...')
            egi_events = _read_events(fid, egi_info)  # update info + jump
            if egi_info['value_range'] != 0 and egi_info['bits'] != 0:
                cal = egi_info['value_range'] / 2 ** egi_info['bits']
            else:
                cal = 1e-6

        logger.info('    Assembling measurement info ...')

        if egi_info['n_events'] > 0:
            event_codes = list(egi_info['event_codes'])
            if include is None:
                exclude_list = ['sync', 'TREV'] if exclude is None else exclude
                exclude_inds = [i for i, k in enumerate(event_codes) if k in
                                exclude_list]
                more_excludes = []
                if exclude is None:
                    for ii, event in enumerate(egi_events):
                        if event.sum() <= 1 and event_codes[ii]:
                            more_excludes.append(ii)
                if len(exclude_inds) + len(more_excludes) == len(event_codes):
                    warn('Did not find any event code with more than one '
                         'event.', RuntimeWarning)
                else:
                    exclude_inds.extend(more_excludes)

                exclude_inds.sort()
                include_ = [i for i in np.arange(egi_info['n_events']) if
                            i not in exclude_inds]
                include_names = [k for i, k in enumerate(event_codes)
                                 if i in include_]
            else:
                include_ = [i for i, k in enumerate(event_codes)
                            if k in include]
                include_names = include

            for kk, v in [('include', include_names), ('exclude', exclude)]:
                if isinstance(v, list):
                    for k in v:
                        if k not in event_codes:
                            raise ValueError('Could find event named "%s"' % k)
                elif v is not None:
                    raise ValueError('`%s` must be None or of type list' % kk)

            event_ids = np.arange(len(include_)) + 1
            logger.info('    Synthesizing trigger channel "STI 014" ...')
            logger.info('    Excluding events {%s} ...' %
                        ", ".join([k for i, k in enumerate(event_codes)
                                   if i not in include_]))
            self._new_trigger = _combine_triggers(egi_events[include_],
                                                  remapping=event_ids)
            self.event_id = dict(zip([e for e in event_codes if e in
                                      include_names], event_ids))
        else:
            # No events
            self.event_id = None
            self._new_trigger = None
        info = _empty_info(egi_info['samp_rate'])
        info['buffer_size_sec'] = 1.  # reasonable default
        my_time = datetime.datetime(
            egi_info['year'], egi_info['month'], egi_info['day'],
            egi_info['hour'], egi_info['minute'], egi_info['second'])
        my_timestamp = time.mktime(my_time.timetuple())
        info['meas_date'] = np.array([my_timestamp], dtype=np.float32)
        ch_names = ['EEG %03d' % (i + 1) for i in
                    range(egi_info['n_channels'])]
        ch_names.extend(list(egi_info['event_codes']))
        if self._new_trigger is not None:
            ch_names.append('STI 014')  # our new_trigger
        nchan = len(ch_names)
        cals = np.repeat(cal, nchan)
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, (), (), misc)
        sti_ch_idx = [i for i, name in enumerate(ch_names) if
                      name.startswith('STI') or len(name) == 4]
        for idx in sti_ch_idx:
            chs[idx].update({'unit_mul': 0, 'cal': 1,
                             'kind': FIFF.FIFFV_STIM_CH,
                             'coil_type': FIFF.FIFFV_COIL_NONE,
                             'unit': FIFF.FIFF_UNIT_NONE})
        info['chs'] = chs
        info._update_redundant()
        _check_update_montage(info, montage)
        super(RawEGI, self).__init__(
            info, preload, orig_format=egi_info['orig_format'],
            filenames=[input_fname], last_samps=[egi_info['n_samples'] - 1],
            raw_extras=[egi_info], verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        egi_info = self._raw_extras[fi]
        dtype = egi_info['dtype']
        n_chan_read = egi_info['n_channels'] + egi_info['n_events']
        offset = 36 + egi_info['n_events'] * 4
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult,
                            dtype=dtype, n_channels=n_chan_read, offset=offset,
                            trigger_ch=self._new_trigger)


@verbose
def read_mff_egi(input_fname, montage=None, eog=None, misc=None, include=None,
                 exclude=None, preload=False, verbose=None):
    """Read EGI Metafile Format as raw object.

    .. note:: The trigger channel names are based on the
              arbitrary user dependent event codes used. However this
              function will attempt to generate a synthetic trigger channel
              named ``STI 014`` in accordance with the general
              Neuromag / MNE naming pattern.

              The event_id assignment equals
              ``np.arange(n_events - n_excluded) + 1``. The resulting
              `event_id` mapping is stored as attribute to the resulting
              raw object but will be ignored when saving to a fiff.
              Note. The trigger channel is artificially constructed based
              on timestamps received by the Netstation. As a consequence,
              triggers have only short durations.

              This step will fail if events are not mutually exclusive.

    Parameters
    ----------
    input_fname : str
        Path to the raw file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). If 'default', will load sensor
        location information from embedded coordinates.xml file. See the
        documentation of :func:`mne.channels.read_montage` for more
        information.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Default is None.
    include : None | list
       The event channels to be ignored when creating the synthetic
       trigger. Defaults to None.
       Note. Overrides `exclude` parameter.
    exclude : None | list
       The event channels to be ignored when creating the synthetic
       trigger. Defaults to None. If None, channels that have more than
       one event and the ``sync`` and ``TREV`` channels will be
       ignored.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).

        ..versionadded:: 0.11

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : Instance of RawEGI
        A Raw object containing EGI data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawMffEGI(input_fname, montage, eog, misc, include, exclude,
                     preload, verbose)


def _check_valid_mff_file(mff_fname):
    """Checks if the supplied mff file is valid."""
    if mff_fname[-4:] != ".mff":
        raise ValueError("Invalid file extension.")
    if not op.exists(mff_fname):
        raise ValueError("Supplied file does not exist")
    if not op.isdir(mff_fname):
        raise ValueError("Invalid MFF file. Supplied file is not a directory.")

    # Check for core necessary files in mff directory
    required_files = ['info.xml', 'sensorLayout.xml', 'signal1.bin']
    fnames = os.listdir(mff_fname)

    for f in required_files:
        if f not in fnames:
            raise ValueError("Invalid MFF file. %s not present." % f)


def _read_recording_info(fname):

    root = ElementTree.parse(fname).getroot()
    ns = root.tag[root.tag.index('{'):root.tag.index('}') + 1]

    # Check for supported MFF Version
    mff_version = int(root.find('%smffVersion' % ns).text)
    if mff_version != 3:
        raise ValueError("Unsupported MFF Version %s" % mff_version)

    # Return the recording start time as posix timestamp
    recording_time = root.find('%srecordTime' % ns)
    return _egi_timestamp_to_posix(recording_time)


def _read_signal_file_header(signal_fname):
    block_info = []
    with open(signal_fname, 'rb') as fid:
        block_info.append(_read_mff_signal_block(fid))
        if 'num_blocks' in block_info[-1].keys():
            for i in range(1, block_info[-1]['num_blocks']):
                block_info.append(_read_mff_signal_block(fid, block_info[-1]))
    return block_info


def _read_mff_signal_block(fid, prev_hdr=None):

    info = dict()

    info['version'] = np.fromfile(fid, '<i4', 1)[0]
    if info['version'] == 0:
        info = prev_hdr
        info['version'] = 0
    else:
        info['headersize'] = np.fromfile(fid, '<i4', 1)[0]
        info['datasize'] = np.fromfile(fid, '<i4', 1)[0]
        info['num_signals'] = np.fromfile(fid, '<i4', 1)[0]
        info['offsets'] = np.fromfile(fid, '<i4', info['num_signals'])

        info['depths'] = []
        info['sfreqs'] = []
        for i in range(info['num_signals']):
            x = np.fromfile(fid, '<i4', 1)[0]
            info['depths'].append(x & 2**8 - 1)
            info['sfreqs'].append(x >> 8)

        info['optlen'] = np.fromfile(fid, '<i4', 1)[0]
        if info['optlen'] > 0:
            info['egi_type'] = np.fromfile(fid, '<i4', 1)[0]
            info['num_blocks'] = np.fromfile(fid, '<i8', 1)[0]
            info['total_samples'] = np.fromfile(fid, '<i8', 1)[0]
            np.fromfile(fid, '<i4', 1)[0]

        info['offsets'] = np.append(info['offsets'], info['datasize'])
        info['num_samples'] = 8 * np.diff(info['offsets']) / \
            np.array(info['depths'])

    # Skip the actual data
    fid.seek(info['datasize'], 1)

    return info


def _read_sensor_names(fname, sensor_type):

    root = ElementTree.parse(fname).getroot()
    ns = root.tag[root.tag.index('{'):root.tag.index('}') + 1]

    ch_names = []

    sensors = root.find("%ssensors")
    for sensor in sensors:
        if sensor_type == 'eeg':
            typ = int(sensor.find("%stype" % ns).text)
            if typ == 0 or typ == 1:
                name = 'EEG %03d' % int(sensor.find("%snumber" % ns).text)
        else:
            name = sensor.find('%sname' % ns).text

        ch_names.append(name)

    return ch_names


def _micros_to_samples(event_time_in_micros, samp_rate):
    microsecs = np.float64(event_time_in_micros)
    samp_duration = 1e6 / samp_rate
    event_time_in_samples = microsecs / samp_duration
    event_time_in_samples = np.fix(event_time_in_samples)
    return event_time_in_samples


def _egi_timestamp_to_posix(timestamp):
    """ Converts an EGI timestamp into an MNE pair of POSIX timestamps
        with millisecond & microsecond resolution. """

    # Construct DateTime Object from EGI timestamp
    year = int(timestamp[0:4])
    month = int(timestamp[5:7])
    day = int(timestamp[8:10])
    hour = int(timestamp[11:13])
    minute = int(timestamp[14:16])
    second = int(timestamp[17:19])
    micro = int(timestamp[20:26])

    time = datetime.datetime(year=year, month=month, day=day, hour=hour,
                             minute=minute, second=second, microsecond=micro)

    # Convert Datetime object into POSIX millisecond & microsecond timestamp
    seconds = time.mktime(time.timetuple())
    microseconds = time.microsecond

    return [seconds, microseconds]


def _read_mff_events(fname, recording_time, sfreq, num_samples):

    root = ElementTree.parse(fname).getroot()
    ns = root.tag[root.tag.index('{'):root.tag.index('}') + 1]

    events = {}
    for child in root:
        if child.tag[-5:] == 'event':
            timestamp = root.find('%sbeginTime' % ns).text
            channel = root.find('%slabel' % ns).text

            milli, micro = _egi_timestamp_to_posix(timestamp)
            event_base_time = milli * 1e6 + micro
            event_time_in_micros = event_base_time - recording_time
            event_time_in_samples = _micros_to_samples(event_time_in_micros,
                                                       sfreq)
            if channel in events:
                events[channel]['samples'].append(event_time_in_samples)
            else:
                events[channel] = {}
                events[channel]['samples'] = [event_time_in_samples]

    event_ch_names = sorted(events.keys())
    stim_chs = np.zeros(len(event_ch_names), num_samples)

    for i, ch in enumerate(event_ch_names):
        stim_chs[i, events[ch]] = 1
    return stim_chs, event_ch_names


class RawMffEGI(BaseRaw):
    """Raw object from EGI Metafile format file."""

    @verbose
    def __init__(self, input_fname, montage=None, dig_montage=None,
                 eog=None, misc=None, include=None, exclude=None,
                 preload=False, verbose=None):  # noqa: D102

        if eog is None:
            eog = []
        if misc is None:
            misc = []

        _check_valid_mff_file(input_fname)

        fnames = os.listdir(input_fname)

        # Extract Header Information from Signal Files
        signal_fnames = [f for f in fnames if 'signal' in f]
        signals_info = []
        for f in signal_fnames:
            signals_info.append(_read_signal_file_header("%s/signal1.bin" %
                                                         input_fname))

        sfreqs = [s[0]['sfreqs'][0] for s in signals_info]
        num_samples = [s[0]['num_samples'][0] for s in signals_info]

        if len(np.unique(sfreqs)) > 1 or len(np.unique(num_samples)) > 1:
            raise ValueError('All recordings must have same sampling ' +
                             'frequency and number of samples.')

        num_samples = int(num_samples[0])
        sfreq = sfreqs[0]

        # Create Info Object
        info = _empty_info(sfreq)
        info['meas_time'] = _read_recording_info('%s/info.xml' % input_fname)
        info['buffer_size_sec'] = 1.  # reasonable default

        # Extract Channels Information

        num_channels = np.sum([s[0]['num_signals'][0] for s in signals_info])

        # Extract EEG Information
        ch_names = _read_sensor_names('%s/sensorLayout.xml' %
                                      input_fname, 'eeg')
        cals = _read_eeg_cals('%s/info1.xml' % input_fname)

        # Extract Physio Information
        if 'pnsSet.xml' in fnames:
            misc_ch_names = _read_sensor_names('%s/pnsSet.xml' % input_fname,
                                               'misc')
            misc = list(set(misc_ch_names) | set(misc))
            ch_names += misc_ch_names

        if num_channels != len(ch_names):
            raise ValueError("Only one physio is supported.")

        # Extract Events & Add Trigger Channels
        if 'Events_8 DINs.xml' in fnames:
            stim_chs, event_codes = _read_mff_events('%s/Events_8 DINs.xml' %
                                                     input_fname,
                                                     info['meas_time'], sfreq,
                                                     num_samples)

            if include is None:
                exclude_list = ['sync', 'TREV'] if exclude is None else exclude
                exclude_inds = [i for i, k in enumerate(event_codes) if k in
                                exclude_list]
                more_excludes = []
                if exclude is None:
                    for ii, event in enumerate(stim_chs):
                        if event.sum() <= 1 and stim_chs[ii]:
                            more_excludes.append(ii)
                if len(exclude_inds) + len(more_excludes) == \
                   len(event_codes):
                    warn('Did not find any event code with more than one '
                         'event.', RuntimeWarning)
                else:
                    exclude_inds.extend(more_excludes)

                exclude_inds.sort()
                include_ = [i for i in np.arange(event_codes) if
                            i not in exclude_inds]
                include_names = [k for i, k in enumerate(event_codes)
                                 if i in include_]
            else:
                include_ = [i for i, k in enumerate(event_codes)
                            if k in include]
                include_names = include

            for kk, v in [('include', include_names), ('exclude', exclude)]:
                if isinstance(v, list):
                    for k in v:
                        if k not in event_codes:
                            raise ValueError('Could find event named "%s"' % k)
                elif v is not None:
                    raise ValueError('`%s` must be None or of type list' % kk)

            event_ids = np.arange(len(include_)) + 1
            logger.info('    Synthesizing trigger channel "STI 014" ...')
            logger.info('    Excluding events {%s} ...' %
                        ", ".join([k for i, k in enumerate(event_codes)
                                   if i not in include_]))
            self._new_trigger = _combine_triggers(stim_chs[include_],
                                                  remapping=event_ids)
            self.event_id = dict(zip([e for e in event_codes if e in
                                      include_names], event_ids))

            ch_names += event_codes
            ch_names.append('STI 014')
            event_codes.append('STI 014')

        else:
            self._new_trigger = None
            self.event_id = None

        # Create Channels Info
        num_channels = len(ch_names)
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, (), (), misc)
        sti_ch_idx = [i for i, name in enumerate(ch_names) if
                      name in event_codes]
        for idx in sti_ch_idx:
            chs[idx].update({'unit_mul': 0, 'cal': 1,
                             'kind': FIFF.FIFFV_STIM_CH,
                             'coil_type': FIFF.FIFFV_COIL_NONE,
                             'unit': FIFF.FIFF_UNIT_NONE})
        info['chs'] = chs
        info._update_redundant()

        _check_update_montage(info, montage)

        super(RawEGI, self).__init__(
            info, preload, orig_format=egi_info['orig_format'],
            filenames=[input_fname], last_samps=[egi_info['n_samples'] - 1],
            raw_extras=[egi_info], verbose=verbose)












    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        egi_info = self._raw_extras[fi]
        dtype = egi_info['dtype']
        n_chan_read = egi_info['n_channels'] + egi_info['n_events']
        offset = 36 + egi_info['n_events'] * 4
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult,
                            dtype=dtype, n_channels=n_chan_read, offset=offset,
                            trigger_ch=self._new_trigger)
