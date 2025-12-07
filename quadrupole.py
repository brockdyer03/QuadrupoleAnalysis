from __future__ import annotations
import numpy as np
import numpy.typing as npt
from os import PathLike


# Helper functions for converting between atomic symbol and atomic number as well as getting atomic masses
def get_atomic_id(atomic_id: str | int) -> int | str:
    """Convert between atomic symbol and atomic number (case insensitive)."""
    elements = {
        "H" : 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B" : 5,
        "C" : 6,
        "N" : 7,
        "O" : 8,
        "F" : 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P" : 15,
        "S" : 16,
        "Cl": 17,
        "Ar": 18,
        "K" : 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V" : 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y" : 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I" : 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Tb": 65,
        "Dy": 66,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W" : 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
        "Fr": 87,
        "Ra": 88,
        "Ac": 89,
        "Th": 90,
        "Pa": 91,
        "U" : 92,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Bk": 97,
        "Cf": 98,
        "Es": 99,
        "Fm": 100,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Rf": 104,
        "Db": 105,
        "Sg": 106,
        "Bh": 107,
        "Hs": 108,
        "Mt": 109,
        "Ds": 110,
        "Rg": 111,
        "Cn": 112,
        "Nh": 113,
        "Fl": 114,
        "Mc": 115,
        "Lv": 116,
        "Ts": 117,
        "Og": 118,
    }

    if isinstance(atomic_id, str):
        return elements[atomic_id.title()]
    if isinstance(atomic_id, int):
        return (list(elements.keys())[list(elements.values()).index(atomic_id)]).title()
    

def get_atomic_mass(atomic_id: str | int):
    """Get atomic mass for an element, specified by either atomic symbol or atomic number (case insensitive)."""

    if isinstance(atomic_id, int):
        atomic_id = get_atomic_id(atomic_id)

    elements = {
        "H"  : 1.0080,
        "He" : 4.002602,
        "Li" : 6.94,
        "Be" : 9.0121831,
        "B"  : 10.81,
        "C"  : 12.011,
        "N"  : 14.007,
        "O"  : 15.999,
        "F"  : 18.998403162,
        "Ne" : 20.1797,
        "Na" : 22.98976928,
        "Mg" : 24.305,
        "Al" : 26.9815384,
        "Si" : 28.085,
        "P"  : 30.973761998,
        "S"  : 32.06,
        "Cl" : 35.45,
        "Ar" : 39.95,
        "K"  : 39.0983,
        "Ca" : 40.078,
        "Sc" : 44.955907,
        "Ti" : 47.867,
        "V"  : 50.9415,
        "Cr" : 51.9961,
        "Mn" : 54.938043,
        "Fe" : 55.845,
        "Co" : 58.933194,
        "Ni" : 58.6934,
        "Cu" : 63.546,
        "Zn" : 65.38,
        "Ga" : 69.723,
        "Ge" : 72.630,
        "As" : 74.921595,
        "Se" : 78.971,
        "Br" : 79.904,
        "Kr" : 83.798,
        "Rb" : 85.4678,
        "Sr" : 87.62,
        "Y"  : 88.905838,
        "Zr" : 91.222,
        "Nb" : 92.90637,
        "Mo" : 95.95,
        "Tc" : 97.,
        "Ru" : 101.07,
        "Rh" : 102.90549,
        "Pd" : 106.42,
        "Ag" : 107.8682,
        "Cd" : 112.414,
        "In" : 114.818,
        "Sn" : 118.710,
        "Sb" : 121.760,
        "Te" : 127.60,
        "I"  : 126.90447,
        "Xe" : 131.293,
        "Cs" : 132.90545196,
        "Ba" : 137.327,
        "La" : 138.90547,
        "Ce" : 140.116,
        "Pr" : 140.90766,
        "Nd" : 144.242,
        "Pm" : 145.,
        "Sm" : 150.36,
        "Eu" : 151.964,
        "Gd" : 157.249,
        "Tb" : 158.925354,
        "Dy" : 162.500,
        "Ho" : 164.930329,
        "Er" : 167.259,
        "Tm" : 168.934219,
        "Yb" : 173.045,
        "Lu" : 174.96669,
        "Hf" : 178.486,
        "Ta" : 180.94788,
        "W"  : 183.84,
        "Re" : 186.207,
        "Os" : 190.23,
        "Ir" : 192.217,
        "Pt" : 195.084,
        "Au" : 196.966570,
        "Hg" : 200.592,
        "Tl" : 204.38,
        "Pb" : 207.2,
        "Bi" : 208.98040,
        "Po" : 209.,
        "At" : 210.,
        "Rn" : 222.,
        "Fr" : 223.,
        "Ra" : 226.,
        "Ac" : 227.,
        "Th" : 232.0377,
        "Pa" : 231.03588,
        "U"  : 238.02891,
        "Np" : 237.,
        "Pu" : 244.,
        "Am" : 243.,
        "Cm" : 247.,
        "Bk" : 247.,
        "Cf" : 251.,
        "Es" : 252.,
        "Fm" : 257.,
        "Md" : 258.,
        "No" : 259.,
        "Lr" : 262.,
        "Rf" : 267.,
        "Db" : 270.,
        "Sg" : 269.,
        "Bh" : 270.,
        "Hs" : 270.,
        "Mt" : 278.,
        "Ds" : 281.,
        "Rg" : 281.,
        "Cn" : 285.,
        "Nh" : 286.,
        "Fl" : 289.,
        "Mc" : 289.,
        "Lv" : 293.,
        "Ts" : 293.,
        "Og" : 294.,
    }

    return elements[atomic_id.title()]


class Atom:
    """Class containing the information of a single atom.

    Attributes
    ----------
    element : str
        The atomic symbol of the element.
    xyz : NDArray
        The x-, y-, and z-coordinates of the atom.
    """

    def __init__(
        self,
        element: str,
        xyz: npt.ArrayLike,
    ):
        self.element = element
        self.xyz = np.array(xyz, dtype=np.float64)

    def __repr__(self):
        return (
            f"{"Element":12}" f"{"X":11}" f"{"Y":11}" f"{"Z":11}\n"
            f"{self.element:9}"f"{self.xyz[0]:11.6f}"f"{self.xyz[1]:11.6f}"f"{self.xyz[2]:11.6f}\n"
        )


class Geometry:
    """Class storing the geometric parameters of a molecular or crystalline geometry.
    
    Attributes
    ----------
    atoms : list[Atom]
        The atoms in the geometry.
    lat_vec : npt.NDArray or None, default=None
        The primitive lattice vectors of the geometry in units of alat
    alat : float or None, default=None
        The lattice parameter.

    Notes
    -----
    The lattice vectors should be provided in units of alat here, which involves taking the square
    root of the sum of the first row of the lattice vector matrix.
    """

    def __init__(
        self,
        atoms: list[Atom],
        lat_vec: npt.NDArray | None = None,
        alat: float | None = None,
    ):
        self.atoms = atoms
        self.lat_vec = lat_vec
        self.alat = float(alat) if alat is not None else None


    def get_coords(self) -> npt.NDArray:
        return np.array([i.xyz for i in self.atoms])


    def get_elements(self) -> list[str]:
        return [i.element for i in self.atoms]


    def remove_atom(self, index: int):
        del self.atoms[index]


    def pop_atom(self, index: int) -> Atom:
        return self.atoms.pop(index)


    def add_atom(self, atom: Atom, index: int = None):
        if index is not None:
            self.atoms.insert(index, atom)
        else:
            self.atoms.append(atom)


    @classmethod
    def from_xsf(cls, file: PathLike) -> Geometry:
        """Read in only the crystallographic information from an XSF file."""
        with open(file, "r") as xsf:

            # Pulls in the lines that contain the primitive lattice vectors and the line containing the number of atoms.
            crystal_info = [next(xsf) for _ in range(7)]

            # Extract the lattice vectors
            lat_vec = np.array([line.strip().split() for line in crystal_info[2:5]], dtype=np.float64)

            # Calculate lattice parameter
            alat = np.sqrt(np.sum(lat_vec[0,:] ** 2))

            lat_vec = lat_vec / alat

            # Pull the number of atoms
            num_atoms = int(crystal_info[-1].split()[0])

            # Read in all of the atoms and turn it into a list of Atom objects
            atoms = [next(xsf).strip().split() for _ in range(num_atoms)]
            atoms = [Atom(element=atom[0], xyz=np.array([float(i) for i in atom[1:4]])) for atom in atoms]

        return Geometry(atoms, lat_vec, alat)


    @classmethod
    def from_xyz(cls, file: PathLike) -> Geometry:
        """Read in XYZ file and return a `Geometry` object"""

        molecule_xyz = []

        with open(file) as xyz:
            for line in xyz:
                # Takes each line in the file minus the last character, which is just the \n
                line = line[:-1].split()
                molecule_xyz.append(line)

        elements = [str(i[0]) for i in molecule_xyz[2:]]
        xyzs = np.array([[float(j) for j in i[1:]] for i in molecule_xyz[2:]])

        atoms = []

        for i, element in enumerate(elements):
            atoms.append(Atom(element, xyzs[i]))

        return Geometry(atoms)


    @classmethod
    def from_list(cls, elements: list[str], xyzs: npt.NDArray, charge: int = 0):
        if len(elements) != len(xyzs):
            raise ValueError("The list of elements and coordinates must be of the same size!")
        else:
            atoms = []
            for i, element in enumerate(elements):
                atoms.append(Atom(element, xyzs[i]))

        return Geometry(atoms, charge)


    @classmethod
    def from_orca(cls, file: PathLike) -> Geometry:
        xyz_data = []
        with open(file, "r") as orca_out:
            found_xyz = False
            while not found_xyz:
                line = orca_out.readline().strip()
                if line == "CARTESIAN COORDINATES (ANGSTROEM)":
                    orca_out.readline()
                    found_xyz = True

            finished_xyz = False
            while not finished_xyz:
                line = orca_out.readline().strip()
                if line == "":
                    finished_xyz = True
                else:
                    xyz_data.append(line.split())

        atoms = [Atom(i[0], np.array(i[1:4])) for i in xyz_data]
        return Geometry(atoms)


    def calc_principal_moments(self):
        """Calculate the principal inertial axes for a given geometry.
        
        Returns
        -------
        eigenvalues : ndarray
            First output of numpy.linalg.eig(inertia_tensor)
        eigenvectors : ndarray
            Second output of numpy.linalg.eig(inertia_tensor)
        """
        center_of_mass = np.zeros(3, dtype=float)
        total_mass = 0.
        for atom in self:
            mass = get_atomic_mass(atom.element)
            center_of_mass += atom.xyz * mass
            total_mass += mass

        center_of_mass = center_of_mass / total_mass

        inertia_matrix = np.zeros((3, 3), dtype=float)

        for atom in self:
            mass = get_atomic_mass(atom.element)
            x = atom.xyz[0] - center_of_mass[0]
            y = atom.xyz[1] - center_of_mass[1]
            z = atom.xyz[2] - center_of_mass[2]

            xx = mass * (y**2 + z**2)
            yy = mass * (x**2 + z**2)
            zz = mass * (x**2 + y**2)

            xy = mass * (x * y)
            xz = mass * (x * z)
            yz = mass * (y * z)

            inertia_matrix[0,0] += xx
            inertia_matrix[1,1] += yy
            inertia_matrix[2,2] += zz

            inertia_matrix[0,1] += -xy
            inertia_matrix[1,0] += -xy

            inertia_matrix[0,2] += -xz
            inertia_matrix[2,0] += -xz

            inertia_matrix[1,2] += -yz
            inertia_matrix[2,1] += -yz

        eigenvalues, eigenvectors = np.linalg.eig(inertia_matrix)

        return eigenvalues, eigenvectors


    def __repr__(self):
        self_repr = ""
        if self.lat_vec is not None:
            lat_vec_scaled = self.lat_vec * self.alat

            self_repr += f"{"Lattice":12}{"X":11}{"Y":11}{"Z":11}\n{"Vectors":11}\n"
            self_repr += f"{"":9}{lat_vec_scaled[0][0]:11.6f}{lat_vec_scaled[0][1]:11.6f}{lat_vec_scaled[0][2]:11.6f}\n"
            self_repr += f"{"":9}{lat_vec_scaled[1][0]:11.6f}{lat_vec_scaled[1][1]:11.6f}{lat_vec_scaled[1][2]:11.6f}\n"
            self_repr += f"{"":9}{lat_vec_scaled[2][0]:11.6f}{lat_vec_scaled[2][1]:11.6f}{lat_vec_scaled[2][2]:11.6f}\n\n"

            self_repr += f"{"Element":12}{"X":11}{"Y":11}{"Z":11}\n\n"
            for i in self.atoms:
                self_repr += f"{i.element:9}{i.xyz[0]*self.alat:11.6f}{i.xyz[1]*self.alat:11.6f}{i.xyz[2]*self.alat:11.6f}\n"
        else:
            self_repr += f"{"Element":12}{"X":11}{"Y":11}{"Z":11}\n\n"
            for i in self.atoms:
                self_repr += f"{i.element:9}{i.xyz[0]:11.6f}{i.xyz[1]:11.6f}{i.xyz[2]:11.6f}\n"
        return self_repr


    def __iter__(self):
        yield from self.atoms


    def __len__(self):
        return len(self.atoms)


class Quadrupole:
    """Class containing data and functions required for analyzing a quadrupole moment.
    
    Attributes
    ----------
    quadrupole : ndarray
        Array containing the 3x3 quadrupole matrix or the diagonal components of the quadrupole (shape 3x1)
    units : {"au", "Buckingham", "Cm^2", "esu"}, default="Buckingham"
        Units of the quadrupole matrix.
    """

    au_cm2_conversion   = 4.4865515185e-40
    esu_cm2_conversion  = 2.99792458e13
    esu_buck_conversion = 1e-26

    def __init__(self, quadrupole: npt.ArrayLike, units: str = "Buckingham"):
        quadrupole = np.array(quadrupole)
        if quadrupole.shape == (3, 3):
            self.quadrupole = np.array(quadrupole, dtype=float)
        elif quadrupole.shape == (3,):
            self.quadrupole = np.diag(quadrupole).astype(float)
        else:
            raise ValueError(f"Cannot cast array of shape {quadrupole.shape} to a quadrupole, supply either shape (3, 3) or (3,)!")
        if units.lower() not in ["au", "buckingham", "cm^2", "esu"]:
            raise ValueError("Invalid units, please select from ( 'au', 'buckingham', 'cm^2', 'esu' )")
        else:
            self.units = units.lower()


    #-----------------------------------------------------------#
    def au_to_cm2(self) -> Quadrupole:                          #
        r"""Convert from Hartree atomic units to Coulomb\*m^2"""#
        q = self.quadrupole * Quadrupole.au_cm2_conversion      #
        return Quadrupole(q, units="cm^2")                      #
                                                                # https://physics.nist.gov/cgi-bin/cuu/Value?aueqm
    def cm2_to_au(self) -> Quadrupole:                          #
        r"""Convert from Coulomb\*m^2 to Hartree atomic units"""#
        q = self.quadrupole / Quadrupole.au_cm2_conversion      #
        return Quadrupole(q, units="au")                        #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def cm2_to_esu(self) -> Quadrupole:                         #
        r"""Convert from Coulomb\*m^2 to e.s.u\*cm^2"""         #
        q = self.quadrupole * Quadrupole.esu_cm2_conversion     #
        return Quadrupole(q, units="esu")                       # CGS statCoulomb/cm^2 to Coulomb/m^2
                                                                # Factor of c * (100cm)^2/m^2
    def esu_to_cm2(self) -> Quadrupole:                         # c taken from https://physics.nist.gov/cgi-bin/cuu/Value?c
        r"""Convert from e.s.u\*cm^2 to Coulomb\*m^2"""         #
        q = self.quadrupole / Quadrupole.esu_cm2_conversion     #
        return Quadrupole(q, units="cm^2")                      #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def buck_to_esu(self) -> Quadrupole:                        #
        r"""Convert from Buckingham to e.s.u\*cm^2"""           #
        q = self.quadrupole * Quadrupole.esu_buck_conversion    #
        return Quadrupole(q, units="esu")                       # Suggested by Peter J. W. Debye in 1963
                                                                # https://doi.org/10.1021/cen-v041n016.p040
    def esu_to_buck(self) -> Quadrupole:                        #
        r"""Convert from e.s.u\*cm^2 to Buckingham"""           #
        q = self.quadrupole / Quadrupole.esu_buck_conversion    #
        return Quadrupole(q, units="Buckingham")                #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def cm2_to_buck(self) -> Quadrupole:                        #
        r"""Convert from Buckingham to Coulomb\*m^2"""          #
        q = self.cm2_to_esu()                                   #
        return q.esu_to_buck()                                  #
                                                                #
    def buck_to_cm2(self) -> Quadrupole:                        #
        r"""Convert from Coulomb\*m^2 to Buckingham"""          #
        q = self.buck_to_esu()                                  #
        return q.esu_to_cm2()                                   #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def au_to_esu(self) -> Quadrupole:                          #
        r"""Convert from Hartree atomic units to e.s.u\*cm^2""" #
        q = self.au_to_cm2()                                    #
        return q.cm2_to_esu()                                   #
                                                                #
    def esu_to_au(self) -> Quadrupole:                          #
        r"""Convert from Hartree atomic units to e.s.u\*cm^2""" #
        q = self.esu_to_cm2()                                   #
        return q.cm2_to_au()                                    #
    #-----------------------------------------------------------#

    #-----------------------------------------------------------#
    def au_to_buck(self) -> Quadrupole:                         #
        """Convert from Hartree atomic units to Buckingham"""   #
        q = self.au_to_cm2()                                    #
        q = q.cm2_to_esu()                                      #
        return q.esu_to_buck()                                  #
                                                                #
    def buck_to_au(self) -> Quadrupole:                         #
        """Convert from Buckingham to Hartree atomic units"""   #
        q = self.buck_to_esu()                                  #
        q = q.esu_to_cm2()                                      #
        return q.cm2_to_au()                                    #
    #-----------------------------------------------------------#

    def as_unit(self, units: str) -> Quadrupole:
        """Return quadrupole as a specified unit"""
        self_units = self.units
        new_units = units.lower()
        if new_units not in ["au", "buckingham", "cm^2", "esu"]:
            raise ValueError(f"Unit {units} not recognized, please select from ( 'au', 'buckingham', 'cm^2', 'esu' )")

        if self_units == new_units:
            return self

        if self_units == "buckingham":
            if new_units == "au":
                return self.buck_to_au()
            elif new_units == "cm^2":
                return self.buck_to_cm2()
            elif new_units == "esu":
                return self.buck_to_esu()
        elif self_units == "au":
            if new_units == "buck":
                return self.au_to_buck()
            elif new_units == "cm^2":
                return self.au_to_cm2()
            elif new_units == "esu":
                return self.au_to_esu()
        elif self_units == "esu":
            if new_units == "buck":
                return self.esu_to_buck()
            elif new_units == "cm^2":
                return self.esu_to_cm2()
            elif new_units == "au":
                return self.esu_to_au()
        elif self_units == "cm^2":
            if new_units == "buck":
                return self.cm2_to_buck()
            elif new_units == "au":
                return self.cm2_to_au()
            elif new_units == "esu":
                return self.cm2_to_esu()


    @classmethod
    def from_orca(cls, file: PathLike) -> tuple[Quadrupole]:
        """Read an ORCA output and pull the quadrupole moment(s) from it.

        Returns
        -------
        quad_matrices : tuple[Quadrupole]
            List containing quadrupoles. See Notes for explanation of why
            this can return multiple matrices instead of just one.

        Note
        ----
        For ORCA outputs with methods that produce multiple densities (post-HF methods usually),
        there can be multiple quadrupoles listed. In this case it is up to the user to pull the correct
        quadrupole from the output. Typically, whichever is listed last is the one that is the most accurate
        to the given level of theory of the calculation, but this should be double checked.
        """

        with open(file, "r") as output:
            quadrupoles = []
            for line in output:
                if line.strip().endswith("(Buckingham)"):
                    quadrupoles.append(line.strip().split()[:-1])

        quad_matrices = []
        for quad in quadrupoles[::2]:
            quad_matrix = np.array(
                [
                    [quad[0], quad[3], quad[4]],
                    [quad[3], quad[1], quad[5]],
                    [quad[4], quad[5], quad[2]],
                ], dtype=np.float64
            )
            quad_matrices.append(quad_matrix)

        quads = tuple(Quadrupole(quad, units="Buckingham") for quad in quad_matrices)

        return quads


    def inertialize(self, geometry: Geometry) -> Quadrupole:
        """Rotate the quadrupole into the inertial frame of the given molecular geometry."""
        eigenvalues, eigenvectors = geometry.calc_principal_moments()
        q = np.real_if_close(np.linalg.inv(eigenvectors) @ self.quadrupole @ eigenvectors, tol=1e-8)
        return Quadrupole(q, units=self.units)


    def detrace(self) -> Quadrupole:
        """Apply detracing operation to a quadrupole

        Notes
        -----
        This detracing operator subtracts out the average of the trace of the quadrupole matrix, then multiplies by 3/2.
        The factor of 3/2 comes from the definition of the traceless quadrupole moment from
        Buckingham (1959) (https://doi.org/10.1039/QR9591300183). This is also the form that the NIST CCCBDB
        reports quadrupole moments in.

        It is also important to note that while this is a common definition, there are arguments both for and against the use of the
        traceless quadrupole moment. See https://doi.org/10.1080/00268977500101151 for further discussion.

        ORCA uses a similar definition but instead uses a factor of 3 instead of 3/2.
        Quantum ESPRESSO does not detrace the quadrupole moment.
        """
        q = (3 / 2) * (self.quadrupole - (np.eye(3,3) * (np.trace(self.quadrupole) / 3)))
        return Quadrupole(q, units=self.units)


    def compare(self, expt: Quadrupole):
        """Attempt to align a diagonal calculated quadrupole moment with an experimental quadrupole moment.
        
        Note
        ----
        This code does not guarantee a correct comparison, it simply uses statistical analysis to attempt to
        rotate a calculated quadrupole into the correct frame to be compared to an experimental quadrupole.
        """

        calc_quad = np.diag(self.quadrupole)
        expt_quad = np.diag(expt.quadrupole)

        expt_signs = np.sign(expt_quad)
        calc_signs = np.sign(calc_quad)

        if expt_signs.sum() != calc_signs.sum():
            calc_quad = calc_quad * np.array([-1., -1., -1.])

        abc = np.array([calc_quad[0], calc_quad[1], calc_quad[2]])
        acb = np.array([calc_quad[0], calc_quad[2], calc_quad[1]])
        cba = np.array([calc_quad[2], calc_quad[1], calc_quad[0]])
        cab = np.array([calc_quad[2], calc_quad[0], calc_quad[1]])
        bac = np.array([calc_quad[1], calc_quad[0], calc_quad[2]])
        bca = np.array([calc_quad[1], calc_quad[2], calc_quad[0]])

        permutations = [
            abc,
            acb,
            cba,
            cab,
            bac,
            bca,
        ]

        diffs = []
        for perm in permutations:
            diffs.append([perm, perm - np.array(expt_quad)])

        diffs.sort(key=lambda x: np.std(x[1]))

        best_match = min(diffs, key=lambda x: np.sum(np.abs(x[1])))

        best_quad = best_match[0]

        return Quadrupole(best_quad)


    def __repr__(self):
        quad = self.quadrupole
        self_str  = ""
        if self.units in ["buckingham", "au"]:
            self_str += f"{"Quadrupole Moment":18}({self.units}):      {"(xx)":10} {"(yy)":10} {"(zz)":10} {"(xy)":10} {"(xz)":10} {"(yz)":10}\n"
            self_str += f"{"":15}{" "*len(self.units)}Total: {quad[0,0]:10.5f} {quad[1,1]:10.5f} {quad[2,2]:10.5f} {quad[0,1]:10.5f} {quad[0,2]:10.5f} {quad[1,2]:10.5f}\n"
        else:
            self_str += f"{"Quadrupole Moment":18}({self.units}):      {"(xx)":13} {"(yy)":13} {"(zz)":13} {"(xy)":13} {"(xz)":13} {"(yz)":13}\n"
            self_str += f"{"":15}{" "*len(self.units)}Total: {quad[0,0]:13.5e} {quad[1,1]:13.5e} {quad[2,2]:13.5e} {quad[0,1]:13.5e} {quad[0,2]:13.5e} {quad[1,2]:13.5e}\n"
        return self_str
    

    def __add__(self, quad: Quadrupole) -> Quadrupole:
        q1 = self.quadrupole
        q2 = quad.as_unit(self.units)
        q2 = q2.quadrupole
        return Quadrupole(quadrupole=q1+q2, units=self.units)
    

    def __sub__(self, quad: Quadrupole) -> Quadrupole:
        q1 = self.quadrupole
        q2 = quad.as_unit(self.units)
        q2 = q2.quadrupole
        return Quadrupole(quadrupole=q1-q2, units=self.units)