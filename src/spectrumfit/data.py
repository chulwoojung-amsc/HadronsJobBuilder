import os
import re
import glob
import xml.etree.ElementTree as ET

import numpy as np


def resolve_format(fmt, path, name):
    """Resolve dataset.format, auto-detecting when fmt=='auto'.
    Detection: an explicit .xml in `name`, or any *.out.*.xml under `path`,
    means hadrons_xml; otherwise flat_text."""
    if fmt != 'auto':
        return fmt
    if name.endswith('.xml'):
        return 'hadrons_xml'
    if glob.glob(os.path.join(path, '**', '*.out.*.xml'), recursive=True):
        return 'hadrons_xml'
    return 'flat_text'


def _bin(raw, Nbin):
    """Average consecutive groups of Nbin measurements into one sample.
    raw: (Nmeas, Nt, Nop) with Nmeas divisible by Nbin.
    Returns C of shape (Nmeas//Nbin, Nt, Nop)."""
    Nmeas, Nt, Nop = raw.shape
    assert Nmeas % Nbin == 0, (
        f'Number of measurements {Nmeas} not divisible by Nbin={Nbin}')
    Nsample = Nmeas // Nbin
    C = raw.reshape(Nsample, Nbin, Nt, Nop).mean(axis=1)
    return C


def load_correlators(path, name, Nt, Nbin):
    """Load flat-text correlator data: (Nsample*Nbin*Nt rows) x (Nop cols).
    Consecutive groups of Nbin measurements are averaged into one sample.
    Returns C of shape (Nsample, Nt, Nop)."""
    data = np.genfromtxt(os.path.join(path, name))
    Nop  = data.shape[1]
    assert data.shape[0] % Nt == 0, (
        f'Data length {data.shape[0]} not a multiple of Nt={Nt}')
    Nmeas = data.shape[0] // Nt
    raw = data.reshape(Nmeas, Nt, Nop)
    return _bin(raw, Nbin)


#Hadrons MContraction::Meson output is <base>.out.<traj>.xml, one file per
#trajectory. The trajectory index is the second-to-last dot field.
_XML_RE = re.compile(r'^(?P<base>.+)\.out\.(?P<traj>-?\d+)\.xml$')


def _discover_xml(path, name):
    """Find Hadrons meson XML files under `path` and group by base observable.
    name=='' -> auto-discover across subdirs, require a single base name.
    name!='' -> restrict to that base name.
    Returns (base, [(traj, filepath), ...]) sorted by traj."""
    matches = {}   # base -> list of (traj, filepath)
    for fp in glob.glob(os.path.join(path, '**', '*.out.*.xml'), recursive=True):
        m = _XML_RE.match(os.path.basename(fp))
        if not m:
            continue
        base = m.group('base')
        if name and base != name:
            continue
        matches.setdefault(base, []).append((int(m.group('traj')), fp))

    if not matches:
        where = f'{path!r}' + (f' with base name {name!r}' if name else '')
        raise FileNotFoundError(f'No Hadrons meson XML (*.out.*.xml) found under {where}')
    if len(matches) > 1:
        names = ', '.join(sorted(matches))
        raise ValueError(
            f'Auto-discovery found multiple observables under {path!r}: {names}. '
            f'Set dataset.name to one of them to disambiguate.')

    base, files = next(iter(matches.items()))
    files.sort(key=lambda tf: tf[0])
    return base, files


def _parse_meson_xml(fp):
    """Parse one Hadrons meson XML. Returns (channels, arr) where channels is a
    list of (gamma_snk, gamma_src) label pairs and arr is (Nt, Nop) real."""
    root = ET.parse(fp).getroot()
    elems = root.findall('./meson/elem')
    if not elems:
        raise ValueError(f'{fp}: no <meson>/<elem> blocks found')
    channels, cols = [], []
    for el in elems:
        snk = (el.findtext('gamma_snk') or '').strip()
        src = (el.findtext('gamma_src') or '').strip()
        vals = []
        for c in el.findall('./corr/elem'):
            #entries look like "(re,im)"
            re_str, im_str = c.text.strip().lstrip('(').rstrip(')').split(',')
            vals.append((float(re_str), float(im_str)))
        channels.append((snk, src))
        cols.append(np.array(vals))          # (Nt, 2)
    return channels, np.stack(cols, axis=1)  # (Nt, Nop, 2)


def load_hadrons_xml(path, name, Nt, Nbin=1, imag_tol=1e-6):
    """Load a set of Hadrons meson XML files (one per trajectory) as correlator
    data. Each <meson>/<elem> (gamma_snk, gamma_src) pair becomes one operator
    column; the real part of each <corr> entry is used.

    Returns (C, channels, base) with C of shape (Nsample, Nt, Nop) after Nbin
    binning, channels the list of (gamma_snk, gamma_src) labels (one per column),
    and base the resolved observable name."""
    base, files = _discover_xml(path, name)

    ref_channels = None
    stack = []
    for traj, fp in files:
        channels, arr = _parse_meson_xml(fp)          # (Nt, Nop, 2)
        if ref_channels is None:
            ref_channels = channels
        elif channels != ref_channels:
            raise ValueError(f'{fp}: gamma channels {channels} differ from '
                             f'first file {ref_channels}')
        if arr.shape[0] != Nt:
            raise ValueError(f'{fp}: Nt={arr.shape[0]} does not match configured Nt={Nt}')
        re_part, im_part = arr[..., 0], arr[..., 1]
        denom = np.where(np.abs(re_part) > 0, np.abs(re_part), 1.0)
        max_ratio = np.max(np.abs(im_part) / denom)
        if max_ratio > imag_tol:
            print(f'WARNING {os.path.basename(fp)}: max |im/re| = {max_ratio:.2e} '
                  f'exceeds {imag_tol:.0e}; keeping real part only')
        stack.append(re_part)                         # (Nt, Nop)

    raw = np.stack(stack, axis=0)                      # (Nfiles, Nt, Nop)
    print(f'Hadrons XML: base={base!r}, {len(files)} trajectories '
          f'({files[0][0]}..{files[-1][0]}), {raw.shape[2]} channel(s)')
    return _bin(raw, Nbin), ref_channels, base
