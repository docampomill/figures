import numpy as np

def sorted_alphanumeric(data):
    """
    Function to sort a given of file names in alphanumeric order.
    Author: Daniel Ocampo

    Typical example of use:
        sorted_alphanumeric(os.listdir(dataset_folder))
    """
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_elem_name_by_atomic_num(atomic_numbers):
    ret = []
    if isinstance(atomic_numbers, np.ndarray):
        for entry in atomic_numbers:
            if (entry==1):
                ret.append('H')
            elif (entry==6):
                ret.append('C')
            elif (entry==16):
                ret.append('O')
            elif (entry==42):
                ret.append('Mo')
            elif (entry==52):
                ret.append('Te')
            else:
                raise ValueError(f'Atomic number {entry} is not in the list.')
    
        return np.array(ret)
    else:
        if(atomic_numbers==1):
            return 'H'
        elif (atomic_numbers==6):
            ret.append('C')
        elif (atomic_numbers==16):
            ret.append('O')
        elif (atomic_numbers==42):
            ret.append('Mo')
        elif (atomic_numbers==52):
            ret.append('Te')
        else:
            raise ValueError('Atomic number is not in the list.')


def has_another_line(file):
    cur_pos = file.tell()
    does_it = bool(file.readline())
    file.seek(cur_pos)
    return does_it

        
def moving_average(arr, window_size):
    moving_averages = []
    window_sum = 0

    for i in range(len(arr)):
        window_sum += arr[i]

        if i >= window_size:
            window_sum -= arr[i - window_size]
            moving_averages.append(window_sum / window_size)
        elif i > 0:
            moving_averages.append(window_sum / (i + 1))

    return moving_averages


def exponential_moving_average(arr, alpha):
    ema = []
    ema_prev = arr[0]  # Initialize EMA with the first value of the array

    for i, value in enumerate(arr):
        if i == 0:
            ema.append(ema_prev)
        else:
            ema_curr = alpha * value + (1 - alpha) * ema_prev
            ema.append(ema_curr)
            ema_prev = ema_curr

    return ema


def cumulative_moving_average(arr):
    cum_sum = 0
    cum_averages = []

    for i, value in enumerate(arr, start=1):
        cum_sum += value
        cum_avg = cum_sum / i
        cum_averages.append(cum_avg)

    return cum_averages
        
def read_images(filename, format, path=None, verbose=None, dtype=None):
    """
    Code for reading and visualizing Configuration file format files from N2P2[1]. This code is inspired
    in ASE's lists of atom objects, usually referred to as trajectory files[2].
    Author: Daniel Ocampo

    [1]. https://github.com/CompPhysVienna/n2p2 
         https://compphysvienna.github.io/n2p2/topics/cfg_file.html#cfg-file

    [2]. https://wiki.fysik.dtu.dk/ase/index.html

    Args:
    - filename (String): Could be a wild card (i.e. xyzs/*.xyz)

    returns: calling this function will return a python list filled with Atoms objects. Each instance of 
    the Atoms class is one atomic structure (image) containing the following attributes:
        - lattice (optional)		    - numpy array (floats) with the 9 components of the 3 lattice vectors
        - positions         		    - numpy array (floats) with each atom in the structure has coordinates x, y and z
        - atom types        		    - numpy array (string) with chemical formula of each atom.
        - atom charges (optional)      	    - numpy array (floats) with individual atom charges.
        - atom forces (optional)      	    - numpy array (floats) with fx, fy and fz.
        - img potential energy (optional)   - numpy array (float) with potential energy of atomic structure.
        - img charge (optional)       	    - numpy array (float) with total charge of atomic structure.
    """
    import re
    import ase
    import ase.io
    from os import listdir
    from os.path import join

    if dtype is None:
        dtype = np.float32

    if path is None:
        path='.'
    else:
        filename = join(path, filename)
    
    traj = []
    file = None
    if format == 'n2p2':
        file = open(filename, 'r')

        while (True):
            next_line = file.readline().split()
            if verbose:
                print(next_line)

                if not next_line:
                    break

            current = re.sub('\n', '', next_line[0])
            current = re.sub(r"^\s+", '', current)
            
            img_lattices = []
            img_positions = []
            img_atom_types = []
            img_atom_charges = []
            img_zeros = []
            img_forces = []
            img_pot_energy = []
            img_charge = []
            if current=='begin':
                while not current=='end':
                    current_line = file.readline().split()
                    current = current_line[0]
                    if current:
                        if verbose:
                            print('    ', current)

                        if current_line[0].startswith('comment'):
                            continue
    
                        elif current_line[0].startswith('lattice'):
                            img_lattices.append(current_line[1:])
    
                        elif current_line[0].startswith('atom'):
                            # 2nd, 3rd and 4th columns are cartesian coords of atom
                            img_positions.append([current_line[1], current_line[2], current_line[3]])
                            # 5th column is atom_type
                            img_atom_types.append(current_line[4])
                            # 6th column is atom_charges
                            img_atom_charges.append(current_line[5])
                            # 7th column is not used at the moment
                            img_zeros.append(current_line[6])
                            # 8, 9 and 10 columns are atom forces
                            img_forces.append(current_line[7:])

                        elif current_line[0].startswith('energy'):
                            img_pot_energy.append(current_line[1])

                        elif current_line[0].startswith('charge'):
                            img_charge.append(current_line[1])

                        else:
                            continue

                # image = Atoms(img_lattices, img_positions, img_atom_types, img_atom_charges, img_zeros, img_forces, img_pot_energy, img_charge, dtype=dtype)
                # lattice=None, positions=None, atom_types=None, atom_charges=None, zeros=None,  forces=None, potential_energy=None, charge=None, dtype=None
                image   = Atoms( lattice          = img_lattices,
                                 positions        = img_positions,
                                 atom_types       = img_atom_types,
                                 atom_charges     = img_atom_charges,
                                 zeros            = img_zeros,
                                 forces           = img_forces,
                                 potential_energy = img_pot_energy,
                                 charge           = img_chage,
                                 dtype            = dtype)
                
                traj.append(image)
                continue
    elif(format=='xyz' or format=='extxyz'):
        import glob

        if '*' in filename:
            for fname in sorted_alphanumeric(glob.glob(filename)):
                ase_img = ase.io.read(fname, index=':', format='extxyz', parallel=True)[0]

                forces  = None
                try:
                    forces = ase_img.get_forces()
                except:
                    pass
                
                pot_eng = None
                try:
                    pot_eng = ase_img.get_potential_energy()
                except:
                    pass

                our_img = Atoms(
                              lattice          = ase_img.get_cell().tolist(),
                              positions        = ase_img.get_positions().tolist(),
                              atom_types       = get_elem_name_by_atomic_num(ase_img.get_atomic_numbers()),
                              forces           = forces,
                              potential_energy = pot_eng,
                              dtype            = dtype
                          )
                traj.append(our_img)
                
        else:
            traj = ase.io.read(filename, index=':', format='extxyz', parallel=True)

            for fname in sorted_alphanumeric(glob.glob(filename)):
                ase_img = ase.io.read(fname, index=':', format='extxyz', parallel=True)[0]

                forces  = None
                try:
                    forces = ase_img.get_forces()
                except:
                    pass

                pot_eng = None
                try:
                    pot_eng = ase_img.get_potential_energy()
                except:
                    pass

                our_img = Atoms(
                              lattice           = ase_img.get_cell().tolist(),
                              positions         = ase_img.get_positions().tolist(),
                              atom_types        = get_elem_name_by_atomic_num(ase_img.get_atomic_numbers()),
                              forces            = forces,
                              potential_energy  = pot_eng,
                              dtype             = dtype
                          )
                traj.append(our_img) 
    elif(format=='lammps-dump-text'):
        
        ase_traj = ase.io.read(filename, index=':', format='lammps-dump-text', parallel=True)

        for ase_img in ase_traj:
            forces  = None
            try:
                forces = ase_img.get_forces()
            except:
                pass
            
            pot_eng = None
            try:
                pot_eng = ase_img.get_potential_energy()
            except:
                pass
            
            our_img = Atoms(
                            lattice           = ase_img.get_cell().tolist(),
                            positions         = ase_img.get_positions().tolist(),
		            atom_types        = get_elem_name_by_atomic_num(ase_img.get_atomic_numbers()),
                            forces            = forces,
                            potential_energy  = pot_eng,
                            dtype             = dtype
                      )
            traj.append(our_img)
    elif(format=='cfg'):
        file = open(filename, 'r')

        while (True):
            next_line = file.readline().split()
            if verbose:
                print(next_line)

            if not next_line:
                if not has_another_line(file):
                    break
                continue

            current = re.sub('\n', '', next_line[0])
            current = re.sub(r"^\s+", '', current)

            natoms = None
            img_lattices = []
            img_positions = []
            img_atom_types = []
            img_atom_charges = []
            img_zeros = []
            img_forces = []
            img_pot_energy = []
            img_stress = []
            energy_line_flag = False
            if current.lower()=='begin_cfg':
                while not current.lower()=='end_cfg':
                    current_line = file.readline().split()
                    current = current_line[0]
                    if current:
                        if verbose:
                            print('    ', current, current.isnumeric())

                        if current_line[0].lower().startswith('size'):
                            natoms = file.readline().strip()
                            continue

                        elif current_line[0].lower().startswith('supercell'):
                            for i in range(3):
                                img_lattices.append(file.readline().split())

                        elif current_line[0].lower().startswith('atomdata'):
                            current_line = file.readline().split()
                            current = current_line[0]
                            while current.isnumeric():
                                if current_line[1]=='0':
                                    img_atom_types.append('Mo')
                                elif current_line[1]=='1':
                                    img_atom_types.append('Te')
                                else:
                                    raise ValueError(f'Atom type {current_line[1]} not recognized.')

                                # 3nd, 4rd and 5th columns are cartesian coords of atom
                                img_positions.append([current_line[2], current_line[3], current_line[4]])

                                # 6, 7 and 8 columns are atom forces
                                img_forces.append([current_line[5], current_line[6], current_line[7]])
                                
                                current_line = file.readline().split()
                                current = current_line[0]
                                if current.lower().startswith('energy'):
                                    energy_line_flag = True

                        elif energy_line_flag:
                            img_pot_energy.append(current_line[0].strip())
                            energy_line_flag = False

                        elif current_line[0].lower().startswith('plusstress'):
                            ''' # ==========================================================================
                                # cfg files use eV units. Here, we are converting to eV/Å3
                                # Note: We are changing the sign of the stress!
                                stress_line = file.readline().split()
                                img_lattices_arr = np.array(img_lattices, dtype=dtype)
                                vol = img_lattices_arr[0][0]*img_lattices_arr[1][1]*img_lattices_arr[2][2]
                                stress = -np.array(stress_line, dtype=dtype)/vol
                            ''' # ==========================================================================
                            # cfg files use eV units. Here, we are converting to GPa
                            # Note: We are changing the sign of the stress!
                            stress_line = file.readline().split()
                            img_lattices_arr = np.array(img_lattices, dtype=dtype)
                            vol = img_lattices_arr[0][0]*img_lattices_arr[1][1]*img_lattices_arr[2][2]
                            stress = -(np.array(stress_line, dtype=dtype)/vol)/ase.units.GPa

                            img_stress.append([stress[0], stress[5], stress[4],\
                                               stress[5], stress[1], stress[3],\
                                               stress[4], stress[3], stress[2]])

                # image = Atoms(img_lattices, img_positions, img_atom_types, img_atom_charges, img_zeros, img_forces, img_pot_energy, img_charge, dtype=dtype)
                # lattice=None, positions=None, atom_types=None, atom_charges=None, zeros=None,  forces=None, potential_energy=None, charge=None, dtype=None
                image   = Atoms( lattice          = img_lattices,
                                 positions        = img_positions,
                                 atom_types       = img_atom_types,
                                 forces           = img_forces,
                                 potential_energy = img_pot_energy,
                                 stress           = img_stress,  
                                 dtype            = dtype)
                
                traj.append(image)
                continue
            else:
                l = file.readline()
                if l.strip():
                    continue
                else:
                    break

    
    return traj
                    
        
class Atoms():
    def __init__(self, lattice=None, origin=None, positions=None, atom_types=None, atom_charges=None, zeros=None,  forces=None, potential_energy=None, charge=None, pbc=True, dtype=None):
        """
            origin: (optional) python list (origin can be specified for xyz files only).
        """


        if dtype is None:
            self.dtype = np.float32
        else:
            self.dtype = dtype
        
        self.origin = np.array(origin, dtype=self.dtype)
        self.lattice = np.array(lattice, dtype=self.dtype)
        self.positions = np.array(positions, dtype=self.dtype)
        self.atom_types = get_elem_name_by_atomic_num(np.array(atom_types, dtype=self.dtype))
        self.pbc = pbc


        if not atom_charges is None:
            self.atom_charges = np.array(atom_charges, dtype=self.dtype)

        if not zeros is None:
            self.zeros = np.array(zeros, dtype=self.dtype)

        if not forces is None:
            self.forces = np.array(forces, dtype=self.dtype)

        if not potential_energy is None:
            self.potential_energy = np.array(potential_energy, dtype=self.dtype)

        if not charge is None:
            self.charge = np.array(charge, dtype=self.dtype)


    def read_potential_energies(self, filename):

        file = open(filename, 'r')
        
        potential_energies = []
        stresses = []
        while (True):
            next_line = file.readline().split()
            if not next_line:
                break
                
            if next_line[0].startswith('#'):
                continue
            
            # mdstep? pe pxx pyy pzz pxy pxz pyz
            potential_energies.append(next_line[1])
            stresses.append(next_line[2:])
            
        file.close()
        
        self.potential_energies = np.array(potential_energies, dtype=np.float32)
        self.stresses = np.array(stresses, dtype=np.float32)
        
            
    def set_potential_energies(self, potential_energies):
        self.potential_energies = potential_energies

    def write_to_file(self, filename=None, path=None, format=None):
        from os import mkdir
        from os.path import join,isdir
        
        if path is not None:
            if not isdir(path):
                mkdir(path)
            filename = join(path, filename)

        if format is None:
            raise ValueError('`format` argument must be specified.')

        if format=='n2p2' or format=='N2P2':
            outfile = open(filename, 'w')

            outfile.write('begin\n')

            for i in self.lattice:
                outfile.write(f'lattice  {i[0]:9.6f} {i[1]:9.6f} {i[2]:9.6f}\n')

            for pos,atyp,achrg,zrs,forc in zip(self.positions, self.atom_types, self.atom_charges, self.zeros, self.forces):
                outfile.write(f'atom {pos[0]:12.9f} {pos[1]:12.9f} {pos[2]:12.9f}   {atyp} {achrg:12.9f} {zrs:12.9f}  {forc[0]:12.9f} {forc[1]:12.9f} {forc[2]:12.9f}\n')

            outfile.write(f'energy {self.potential_energy[0]:14.8f}\n')
            outfile.write(f'charge {self.charge[0]:14.8f}\n')
            outfile.write(f'end\n')
            outfile.close()



        elif(format=='xyz'):
            if not filename.endswith('.xyz'):
                filename = filename + '.xyz'


            outfile = open(filename, 'w')

            num_atoms = len(self.positions)
            outfile.write(str(num_atoms)+'\n')

            properties_line = ''
            if not self.lattice.size==0:
                properties_line += 'Lattice="'
                for latt_line in self.lattice:
                    properties_line += f'{latt_line[0]} {latt_line[1]} {latt_line[2]} ' 
                properties_line += '" '
            
            if not self.origin.size==0:
                properties_line += f'Origin="{self.origin[0]} {self.origin[1]} {self.origin[2]}" '

            properties_line += 'Properties=species:S:1:pos:R:3'

            if not self.forces.size==0:
                properties_line += ':forces:R:3 '

            if self.potential_energy:
                properties_line += f'energy={self.potential_energy}'

            if self.pbc:
                properties_line += ' pbc="T T T"'
                
            properties_line += '\n'
            
            atom_lines = ''
            for pos,atyp,forc in zip(self.positions, self.atom_types, self.forces):
                if self.forces.size==0:
                    atom_lines += f'{atyp} {pos[0]:12.9f} {pos[1]:12.9f} {pos[2]:12.9f}'
                else:
                    atom_lines += f'{atyp} {pos[0]:12.9f} {pos[1]:12.9f} {pos[2]:12.9f} {forc[0]:12.9f} {forc[1]:12.9f} {forc[2]:12.9f}'
                atom_lines += '\n'
            

            outfile.write(properties_line)
            outfile.write(atom_lines)
            outfile.close()
            

        elif(format=='lammps-data'):
            import warnings
            # raise ValueError('Exporting to lammps-data format is still not fully implemented.')

            if not filename.endswith('.data'):
                filename = filename + '.data'

            outfile = open(filename, 'w')

            # print('# LAMMPS data file written by GraoGroup\'s AtomDNN 1.0 - Atoms class')
            outfile.write('# LAMMPS data file written by GraoGroup\'s AtomDNN 1.0 - Atoms class\n')
            

            num_atoms = len(self.positions)
            # print(str(num_atoms)+' atoms')
            outfile.write(str(num_atoms)+' atoms\n')

            natom_types = np.unique(self.atom_types).size
            # print(str(natom_types)+ ' atom types')
            outfile.write(str(natom_types)+ ' atom types\n')

            ## lattice
            if not self.lattice.size==0:
                if not (self.lattice[0,1]==0 and self.lattice[0,2]==0):
                    outfile.close()
                    raise ValueError('Conversion to lammps-data lattice is not implemented for non diagonal lattice parameters.')

                if not (self.lattice[1,0]==0 and self.lattice[1,2]==0):
                    outfile.close()
                    raise ValueError('Conversion to lammps-data lattice is not implemented for non diagonal lattice parameters.')

                if not (self.lattice[2,0]==0 and self.lattice[2,1]==0):
                    outfile.close()
                    raise ValueError('Conversion to lammps-data lattice is not implemented for non diagonal lattice parameters.')

                outfile.write(f'0.0 {self.lattice[0,0]:12.7f}   xlo xhi\n')
                outfile.write(f'0.0 {self.lattice[1,1]:12.7f}   ylo yhi\n')
                outfile.write(f'0.0 {self.lattice[2,2]:12.7f}   zlo zhi\n')

                
            ## Atoms
            outfile.write('\nAtoms\n')
            
            if not self.forces.size==0:
                warnings.warn('Forces were ignored when writting lammps-data files.')

            for i,(pos,atype) in enumerate(zip(self.positions, self.atom_types)):            
                outfile.write(f'{i}   {atype} {pos[0]:12.7f} {pos[1]:12.7f} {pos[2]:12.7f}\n')

            outfile.write('\n')
            outfile.close()

            print(f'lammps-data file has been successfully written to {filename}')
            
            #latt_lines = ''
            #if not self.lattice.size==0:
                #for latt_line in self.lattice:
                    #latt_lines += f'{latt_line[0]} {latt_line[1]} {latt_line[2]} '
            # CONVERT 9 PARAMETERS LATTICE SPECIFICATION TO 6 PARAMETERS LATTICE
            
           #0.0 19.1739553015 xlo xhi
           #0.0 10.3094040029 ylo yhi
           #0.0 29.999991153 zlo zhi
            
            
        else:
            raise ValueError('Formats other than N2P2 are not developed for write_to_file() method.')
                
