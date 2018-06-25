from __future__ import print_function, division, unicode_literals, absolute_import

import os
import re
import six
import glob
import numpy as np
import linecache
from copy import copy
from collections import defaultdict
from abipy.core.func1d import Function1D
from pymatgen.electronic_structure.core import Spin, Orbital, OrbitalType
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.abinit.netcdf import structure_from_ncdata, ETSF_Reader
from monty.collections import tree
from monty.io import zopen
from monty.functools import lazy_property


class Coxp(object):
    """
    Warapper class for the crystal orbital projections produced from Lobster.
    Wraps both a COOP and a COHP.
    Can contain both the total and orbitalwise projections.
    """

    def __init__(self, energies, total=None, partial=None, averaged=None, efermi=None):
        self.energies = energies
        self._total_only = total or {}
        self.partial = partial or {}
        self.averaged = averaged or {}
        self.efermi = efermi

        self.total = copy(self._total_only)

        # create the total distribution from the partial
        for index_pair, data in partial.items():
            if index_pair not in total:
                self.total[index_pair] = six.moves.reduce(add_coxp, data.values())
                self.total[(index_pair[1], index_pair[0])] = self.total[index_pair]

    def site_pairs_total(self):
        return [i for i in self.total.keys()]

    def site_pairs_partial(self):
        return [i for i in self.partial.keys()]

    @classmethod
    def from_file(cls, filepath):

        float_patt = r'-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'
        header_patt = re.compile(r'\s+(\d+)\s+(\d+)\s+(\d+)\s+('+float_patt+
                                 r')\s+('+float_patt+r')\s+('+float_patt+r')')
        pair_patt = re.compile(r'No\.\d+:([a-zA-Z]+)(\d+)(?:\[([a-z0-9_\-^]+)\])?->([a-zA-Z]+)(\d+)(?:\[([a-z0-9_\-^]+)\])?')

        with zopen(filepath) as f:
            #find the header
            for line in f:
                match = header_patt.match(line.rstrip())
                if match:
                    n_column_groups = int(match.group(1))
                    n_spin = int(match.group(2))
                    n_en_steps = int(match.group(3))
                    min_en = float(match.group(4))
                    max_en = float(match.group(5))
                    efermi = float(match.group(6))
                    break
            else:
                raise ValueError("Can't find the header in file {}".format(filepath))

            n_pairs = n_column_groups-1

            count_pairs = 0
            pairs_data = []
            #parse the pairs considered
            for line in f:
                match = pair_patt.match(line.rstrip())
                if match:
                    # adds a tuple: [type_1, index_1, orbital_1, type_2, index_2, orbital_2]
                    # with orbital_1, orbital_2 = None if the pair is not orbitalwise
                    type_1, index_1, orbital_1, type_2, index_2, orbital_2 = match.groups()
                    pairs_data.append([type_1, int(index_1), orbital_1, type_2, int(index_2), orbital_2])
                    count_pairs += 1
                    if count_pairs == n_pairs:
                        break

            spins = [Spin.up, Spin.down][:n_spin]

            data = np.fromstring(f.read(), dtype=np.float, sep=' ').reshape([n_en_steps, 1+n_spin*n_column_groups*2])

            # initialize and fill results
            energies = data[:, 0]
            averaged = defaultdict(dict)
            total = tree()
            partial = tree()

            for i, s in enumerate(spins):
                base_index = 1+i*n_column_groups*2
                averaged[s]['single'] = data[:, base_index]
                averaged[s]['integrated'] = data[:, base_index+1]
                for j, p in enumerate(pairs_data):
                    # partial or total
                    if p[2] is not None:
                        single = data[:, base_index+2*(j+1)]
                        integrated = data[:, base_index+2*(j+1)+1]
                        partial[(p[1], p[4])][(p[2], p[5])][s]['single'] = single
                        partial[(p[4], p[1])][(p[5], p[2])][s]['single'] = single
                        partial[(p[1], p[4])][(p[2], p[5])][s]['integrated'] = integrated
                        partial[(p[4], p[1])][(p[5], p[2])][s]['integrated'] = integrated
                    else:
                        single = data[:, base_index+2*(j+1)]
                        integrated = data[:, base_index+2*(j+1)+1]
                        total[(p[1], p[4])][s]['single'] = single
                        total[(p[4], p[1])][s]['single'] = single
                        total[(p[1], p[4])][s]['integrated'] = integrated
                        total[(p[4], p[1])][s]['integrated'] = integrated

        return cls(energies=energies, total=total, partial=partial, averaged=averaged, efermi=efermi)

    @lazy_property
    def get_dos_pair_lorbitals(self):
        if not self.partial:
            raise RuntimeError("Partial orbitals not calculated.")

        results = tree()
        for pair, pair_data in self.partial.items():
            #check if the symmetric has already been calculated
            if (pair[1], pair[0]) in results:
                for orbs, orbs_data in results[(pair[1], pair[0])].items():
                    results[pair][(orbs[1], orbs[0])] = orbs_data
                continue

            #for each look at all orbital possibilities
            for orbs, orbs_data in pair_data.items():
                k=(orbs[0].split("_")[0], orbs[1].split("_")[0])
                if k in results[pair]:
                    for spin in orbs_data.keys():
                        results[pair][k][spin]=results[pair][k][spin]+Function1D(self.energies,orbs_data[spin]['single'])
                else:
                    for spin in orbs_data.keys():
                        results[pair][k][spin]=Function1D(self.energies, orbs_data[spin]['single'])

        return results

    @lazy_property
    def get_dos_pair_morbitals(self):
        if not self.partial:
            raise RuntimeError("Partial orbitals not calculated.")
        results = tree()
        for pair, pair_data in self.partial.items():
            for orbs, orbs_data in pair_data.items():
                for spin in orbs_data.keys():
                    results[pair][orbs][spin]=Function1D(self.energies, orbs_data[spin]['single'])
        return results

    @lazy_property
    def get_dos_pair(self):
        results = tree()
        for pair, pair_data in self.total.items():
            for spin in pair_data.keys():
                results[pair][spin] = Function1D(self.energies, pair_data[spin]['single'])
        return results

    @lazy_property
    def get_partial_lorbitals(self):
        results = tree()
        for pair, pair_data in self.partial.items():
            #check if the symmetric has already been calculated
            if (pair[1], pair[0]) in results:
                for orbs, orbs_data in results[(pair[1], pair[0])].items():
                    results[pair][(orbs[1], orbs[0])] = orbs_data
                continue

            #for each look at all orbital possibilities
            for orbs, orbs_data in pair_data.items():
                k=(orbs[0].split("_")[0], orbs[1].split("_")[0])
                for spin in orbs_data:
                    results[pair][k][spin]['single'] = results[pair][k][spin].get('single', np.zeros(len(orbs_data[spin]['single'])))+orbs_data[spin]['single']
                    results[pair][k][spin]['integrated'] = results[pair][k][spin].get('integrated', np.zeros(len(orbs_data[spin]['integrated'])))+orbs_data[spin]['integrated']

        return results


class ICoxp(object):

    def __init__(self, averages):
        self.averages = averages

    @classmethod
    def from_file(cls, filepath):
        float_patt = r'-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'
        header_patt = re.compile(r'.*?(over+\s#\s+bonds)?\s+for\s+spin\s+(\d).*')
        data_patt = re.compile(r'\s+\d+\s+([a-zA-Z]+)(\d+)\s+([a-zA-Z]+)(\d+)\s+('+
                               float_patt+r')\s+('+float_patt+r')(\d+)?')
        averages = {}
        spin = None
        avg_num_bonds = False
        with zopen(filepath) as f:
            for line in f:
                match = header_patt.match(line.rstrip())
                if match:
                    spin = [Spin.up, Spin.down][int(match.group(2))-1]
                    avg_num_bonds = match.group(1) is not None
                    averages[spin] = defaultdict(dict)
                match = data_patt.match(line.rstrip())
                if match:
                    type_1, index_1, type_2, index_2, dist, avg, n_bonds = match.groups()
                    avg_data = {'average': float(avg), 'distance': dist, 'n_bonds': int(n_bonds) if n_bonds else None}
                    averages[spin][(int(index_1), int(index_2))] = avg_data
                    averages[spin][(int(index_2), int(index_1))] = avg_data

        return cls(averages)


class LobsterInput(object):

    basis_sets = {"bunge", "koga", "pbevaspfit2015"}

    available_advanced_options = {"basisRotation", "writeBasisFunctions", "onlyReadVasprun.xml", "noMemoryMappedFiles",
                                  "skipPAWOrthonormalityTest", "doNotIgnoreExcessiveBands", "doNotUseAbsoluteSpilling",
                                  "skipReOrthonormalization", "doNotOrthogonalizeBasis", "forceV1HMatrix",
                                  "noSymmetryCorrection", "symmetryDetectionPrecision", "useOriginalTetrahedronMethod",
                                  "useDecimalPlaces", "forceEnergyRange"}

    def __init__(self, basis_set=None, basis_functions=None, include_orbitals=None, atom_pairs=None, dist_range=None,
                 orbitalwise=True, start_en=None, end_en=None, en_steps=None, gaussian_smearing=None,
                 bwdf=None, advanced_options=None):
        """
        Args
            basis_set: String containing one of the possible basis sets available: bunge, koga, pbevaspfit2015
            basis_functions: list of strings giving the symbol of each atom and the basis functions: "Ga 4s 4p"
            include_orbitals: string containing which types of valence orbitals to use. E.g. "s p d"
            atom_pairs: list of tuples containing the couple of elements for which the COHP analysis will be
             performed. Index is 1-based.
            dist_range: list of tuples, each containing the minimum and maximum distance (in Angstrom) used to
             automatically generate atom pairs. Each tuple can also contain two atomic symbol to restric the match to
             the specified elements. examples: (0.5, 1.5) or (0.5, 1.5, 'Zn', 'O')
            start_en: starting energy with respect to the Fermi level (in eV)
            end_en: ending energy with respect to the Fermi level (in eV)
            en_steps: number of energy increments
            gaussian_smearing: smearing in eV when using gaussian broadening
            bwdf: enables the bond-weighted distribution function (BWDF). Value is the binning interval
            advanced_options: dict with additional advanced options. See lobster user guide for further details
        """

        if basis_set and basis_set.lower() not in self.basis_sets:
            raise ValueError("Wrong basis set {}".format(basis_set))
        self.basis_set = basis_set
        self.basis_functions = basis_functions or []
        self.include_orbitals = include_orbitals
        self.atom_pairs = atom_pairs or []
        self.dist_range = dist_range or []
        self.orbitalwise = orbitalwise
        self.start_en = start_en
        self.end_en = end_en
        self.en_steps = en_steps
        self.gaussian_smearing = gaussian_smearing
        self.bwdf = bwdf
        self.advanced_options = advanced_options or {}
        if not all(opt in self.available_advanced_options for opt in self.advanced_options.keys()):
            raise ValueError("Unknown adavanced options")

    @classmethod
    def _get_basis_functions_from_abinit_pseudos(cls, pseudos):
        basis_functions = []
        for p in pseudos:
            basis_functions.append(p.symbol + " ".join(str(vs['n']) + OrbitalType(vs['l']).name)
                                   for vs in p.valence_states.values() if 'n' in vs)
            # bf = p.symbol
            # for vs in pseudo.valence_states.values():
            # if 'n' in vs:
            #         bf += " {}{}".format(vs['n'], OrbitalType(vs['l']).name)
            #
            # basis_functions.append(bf)
        return basis_functions

    def set_basis_functions_from_abinit_pseudos(self, pseudos):
        basis_functions = self._get_basis_functions_from_abinit_pseudos(pseudos)

        self.basis_functions = basis_functions

    @classmethod
    def _get_basis_functions_from_potcar(cls, potcars):
        basis_functions = []
        for p in potcars:
            basis_functions.append(p.element +" "+ " ".join(str(vs[0]) + vs[1] for vs in p.electron_configuration))
        return basis_functions

    def set_basis_functions_from_potcar(self, potcar):
        basis_functions = self._get_basis_functions_from_potcar(potcar)

        self.basis_functions = basis_functions

    def __str__(self):
        return self.to_string()

    def to_string(self):
        """
        String representation.
        """
        lines = []

        if self.basis_set:
            lines.append("basisSet "+self.basis_set)

        for bf in self.basis_functions:
            lines.append("basisFunctions "+bf)

        for ap in self.atom_pairs:
            line = "cohpBetween atom {} atom {}".format(*ap)
            if self.orbitalwise:
                line += " orbitalwise"
            lines.append(line)

        for dr in self.dist_range:
            line = "cohpGenerator from {} to {}"+" ".format(dr[0], dr[1])
            if len(dr) > 2:
                line += " type {} type {}".format(dr[2], dr[3])
            if self.orbitalwise:
                line += " orbitalwise"
            lines.append(line)

        if self.start_en:
            lines.append("COHPStartEnergy {}".format(self.start_en))

        if self.end_en:
            lines.append("COHPEndEnergy {}".format(self.end_en))

        if self.en_steps:
             lines.append("COHPSteps {}".format(self.en_steps))

        if self.gaussian_smearing:
            lines.append("gaussianSmearingWidth {}".format(self.gaussing_smearing))

        if self.bwdf:
            lines.append("BWDF {}".format(self.bwdf))

        for k, v in self.advanced_options.items():
            lines.append(k + str(k))

        return "\n".join(lines)

    @classmethod
    def from_run_dir(cls, dirpath, dE=0.01, set_pairs=False, **kwargs):

        # Try to determine the code used for the calculation
        dft_code = None
        if os.path.isfile(os.path.join(dirpath, 'vasprun.xml')):
            dft_code = "vasp"
            vr = Vasprun(os.path.join(dirpath, 'vasprun.xml'))

            en_min = np.min([bands_spin for bands_spin in vr.eigenvalues.values()])
            en_max = np.max([bands_spin for bands_spin in vr.eigenvalues.values()])
            efermi = vr.efermi

            #TODO read potcar (PotcarSingle object). Done?
            potcar = Potcar.from_file('POTCAR')
            basis_functions = cls._get_basis_functions_from_potcar(potcar)

        elif glob.glob(os.path.join(dirpath, '*.files')):
            dft_code = "abinit"
            ff = glob.glob(os.path.join(dirpath, '*.files'))[0]
            out_path = linecache.getline(ff, 4).strip()
            if not os.path.isabs(out_path):
                out_path = os.path.join(dirpath, out_path)

            gsr = ETSF_Reader(out_path+'_GSR.nc')

            eigenvalues = gsr.read_value('eigenvalues')
            en_min = eigenvalues.min()
            en_max = eigenvalues.max()
            efermi = gsr.read_value('fermi_energy')

            basis_functions = cls._get_basis_functions_from_abinit_pseudos()
        else:
            raise ValueError('Unable to determine the code used in dir {}'.format(dirpath))

        start_en = en_min + efermi
        end_en = en_max - efermi

        # shift the energies so that are divisible by dE and the value for the fermi level (0 eV) is included
        start_en = np.floor(start_en/dE)*dE
        end_en= np.ceil(end_en/dE)*dE

        en_steps = (end_en-start_en)/dE

        return cls(basis_functions=basis_functions, start_en=start_en, end_en=end_en, en_steps=en_steps, **kwargs)


    def write_file(self, dirpath='.'):
        """
        Write the input file 'lobsterin' in 'dirpath'.
        """
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # Write the input file.
        with open(os.path.join(dirpath, 'lobsterin'), "wt") as f:
            f.write(str(self))


