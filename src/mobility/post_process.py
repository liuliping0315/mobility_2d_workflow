import os
import sys
import shutil
import subprocess
import time
import pathlib
from typing import List, Tuple

import numpy as np
import numpy.linalg as LA

strain_list = [0.990, 0.995, 1.000, 1.005, 1.010]
strain_str = ["{:1d}_{:.3f}".format(i, j) for j in strain_list for i in [0, 1]]
ratios = [0.990, 0.995, 1.000, 1.005, 1.010]
k_delta = 0.02


def gen_strain(init_dir=None):
    if init_dir == None:
        init_dir = os.getcwd()
    poscar = open("POSCAR", 'r')
    txt = poscar.readlines()
    poscar.close()
    lat_coeff = float(txt[1].split()[0])
    init_cell = np.loadtxt(fname="POSCAR", skiprows=2, max_rows=3)
    init_cell = init_cell * lat_coeff

    for i in [0, 1]:
        for j in range(5):
            my_ratio = ratios[j]
            # this_dir = os.path.join(
            #     "origin", "{:1d}_{:.3f}".format(i, my_ratio))
            this_dir = "{:1d}_{:.3f}".format(i, my_ratio)
            os.makedirs(this_dir, exist_ok=True)
            new_cell = init_cell.copy()
            new_cell[:, i] = init_cell[:, i] * my_ratio
            new_txt = txt.copy()
            new_txt[1] = "1.000\n"
            for k in range(3):
                new_txt[2+k] = "{:.6f} {:.6f} {:.6f}\n".format(new_cell[k, 0], new_cell[k, 1], new_cell[k, 2])

            poscar_name = os.path.join(this_dir, "POSCAR")
            fout = open(poscar_name, 'w')
            fout.writelines(new_txt)
            fout.close()
            shutil.copy("INCAR", this_dir)
            shutil.copy("POTCAR", this_dir)
            shutil.copy("KPOINTS", this_dir)
            shutil.copy("kopt_edge", this_dir+"/KPOINTS_OPT")


def gen_kpoints():
    os.system("vaspkit -task 302")
    os.system("mv KPATH.in KPOINTS_OPT")
    cell = np.loadtxt(fname="POSCAR", dtype=float,
                      skiprows=2, max_rows=3)
    cell_coeff = np.loadtxt("POSCAR", dtype=float, skiprows=1, max_rows=1).squeeze()
    cell = cell * cell_coeff
    lattice_vector_length = LA.norm(cell, axis=1)
    nk1 = 30 // lattice_vector_length[0]
    nk2 = 30 // lattice_vector_length[1]
    nk3 = 1
    if nk1 < 1:
        nk1 = 1
    if nk2 < 1:
        nk2 = 1
    kpt_scf = open("KPOINTS", 'w')
    kpt_scf.write("Automatic mesh\n")
    kpt_scf.write("0\n")
    kpt_scf.write("Gamma\n")
    kpt_scf.write("{:d} {:d} {:d}\n".format(nk1, nk2, nk3))
    kpt_scf.write("0 0 0\n")
    kpt_scf.close()

def get_lvhar():
    locpot = open("LOCPOT", 'r')
    txt = locpot.readlines()
    locpot.close()
    outcar = open("OUTCAR", 'r')
    nx = 0
    ny = 0
    nz = 0
    for line in outcar:
        if "support grid" in line:
            tmp = line.split()
            nx = int(tmp[3])
            ny = int(tmp[5])
            nz = int(tmp[7])
    outcar.close()
    if nx==0 or ny==0 or nz==0:
        print("nx=0?")
        sys.exit(-1)
    ntot = nx * ny * nz
    n_data_lines = int(np.ceil(ntot/5))
    data = txt[-n_data_lines:]
    pot = []
    for line in data:
        pot += [float (tmp) for  tmp in line.split()]
    pot_array = np.array(pot).reshape(nz, ny, nx)
    pot_z = np.sum(pot_array,axis=[1,2])
    return pot_z
    
def calc_mobility():
    # this function is a comment
    # now we have c2d, dfc, m*, m*_d in atomic units as 4-element arrays
    # convert 1Har to J: 1Har = 4.3597447222071e-18 J
    # h_bar in J: h_bar = 1.054571817e-34 J*s
    # Qiao. etc. (Wei Ji), Natrue Communication， DOI: 10.1038/ncomms5475
    # \mu_2D = e * h_bar**3 * C_2D / (k_B*T * m_e * m_d * D_ij**2)
    # \mu_2D units m**2/V/s
    # unit converting, two ways:
    # (1)
    # convert a.u. to SI for original data:
    #               1.602176634e-19 * 1.054571817e-34 ** 3 * 4.3597447222071e-18 / (0.52917721067**2*1e-10**2)
    #  unit_\mu = -------------------------------------------------------------------------------------------- = 4.254382154685228e-06 m**2/V/s
    #               4.3597447222071e-18 * (9.1093837E-31 ** 2) * (4.3597447222071e-18 ** 2)
    # (2)
    # convert a.u. to SI for result:
    #  unit_\mu = 1 Bohr**2/V_har/s_har_ = 0.529177**2 * 1E-10**2 /27.211386245988/ 2.418884 / 1E-17 = 4.2543793405733925e-06 m**2/V/s
    # use (2)
    emass = np.array(op_in["emass"])
    # convert c2d from eV/Angstrom**2 to Hartree/Bohr**2
    # c2d_x, c2d_y -> c2d_x, c2d_y, c2d_x, c2d_y
    c2d = np.array(op_in["c2d"] * 2) * (1/27.211386245988) / (1/0.52917721067)**2
    # convert dfc from eV to Hartree
    dfc = np.array(op_in["dfc"]) / 27.211386245988
    # kT in atomic units
    kT_au = 0.000086173324 * 300 / 27.211386245988
    os.makedirs("result", exist_ok=True)
    mobility = np.zeros((4), dtype=float)
    # md = sqrt(mx * my)
    m_cbm_d = np.sqrt(emass[0]*emass[1])
    m_vbm_d = np.sqrt(emass[2]*emass[3])
    m_d = np.array([m_cbm_d, m_cbm_d, m_vbm_d, m_vbm_d])
    mu_au = 1.0 * 1.0**3 * c2d / (kT_au * emass * m_d * dfc**2)
    mu_si = mu_au * 4.254382154685228e-06
    pass


class VaspOut:
    eigen_array_shifted: np.ndarray

    def __init__(self):
        self.cell = np.zeros((3, 3), dtype=float)
        self.e_tot = 0.0
        self.e_fermi = 0.0
        self.e_vacuum = 0.0
        self.cbm = 0.0
        self.vbm = 0.0
        self.cbm_k_index = 0
        self.vbm_k_index = 0
        self.cbm_b_index = 0
        self.vbm_b_index = 0
        self.get_cell()
        self.get_e_tot()
        self.get_e_fermi()
        self.get_vacumm_level_z()
        self.workdir = os.getcwd()

    def get_cell(self):
        self.cell = np.loadtxt(fname="POSCAR", dtype=float,
                               skiprows=2, max_rows=3)
        cell_coeff = np.loadtxt("POSCAR", dtype=float, skiprows=1, max_rows=1).squeeze()
        self.cell = self.cell * cell_coeff
        self.reclat = LA.inv(self.cell).T

        return self.cell

    def get_e_tot(self):
        with open("OUTCAR", "r") as f:
            lines = f.readlines()
        for line in lines:
            if "energy  without entropy=" in line:
                self.e_tot = float(line.split()[3])

    def get_e_fermi(self):
        with open("OUTCAR", "r") as f:
            lines = f.readlines()
        for line in lines:
            if "E-fermi" in line:
                self.e_fermi = float(line.split()[2])

    def get_vacumm_level_z(self):
        # sep = vaspout.Locpot.from_file("LOCPOT")
        # vz = sep.get_average_along_axis(2)
        # self.e_vaccum = np.max(vz)
        current_dir = os.getcwd()
        os.chdir(self.workdir)
        pot_z = get_lvhar()
        os.chdir(current_dir)
        return pot_z
    def get_band_structure_Kopt(self):
        procar_opt = open("PROCAR_OPT", 'r')
        txt = procar_opt.readlines()
        procar_opt.close()
        metadata = txt[1].split()
        nk = int(metadata[3])
        nb = int(metadata[7])
        kpts_list = []
        all_eigens = []
        # grep "# energy" PROCAR_OPT
        num_lines = len(txt)
        for i in range(num_lines):
            if r"# energy" in txt[i]:
                all_eigens.append(float(txt[i].split()[4]))
            # vasp write PROCAR_OPT with fixed length and NO SPACE between two float number if minus number exists.
            # fortran write format: FORMAT(/' k-point ',I5,' :',3X,3F11.8,' weight = ',F10.8/)
            if r" k-point " in txt[i]:  # important space
                line = txt[i]
                tmp_kpt = [line[19:30], line[30:41], line[41:52]]
                kpts_list.append(tmp_kpt)
        # kpts_list = np.loadtxt("KPOINTS_OPT", skiprows=3, usecols=3)
        eigens_array = np.array(all_eigens)

        eigens_array_shifted = eigens_array - self.e_fermi

        #  deep copy eigens_array_shifted to e1
        e1 = eigens_array_shifted.copy()
        e2 = eigens_array_shifted.copy()

        e1[eigens_array_shifted < 0] = 1000
        e2[eigens_array_shifted > 0] = -1000
        # find the index and value of the min of e1, and max of e2
        cbm_index = np.argmin(e1)
        vbm_index = np.argmax(e2)
        cbm = eigens_array_shifted[cbm_index]
        vbm = eigens_array_shifted[vbm_index]

        cbm_k_index = cbm_index // nb
        vbm_k_index = vbm_index // nb
        cbm_b_index = cbm_index % nb
        vbm_b_index = vbm_index % nb
        self.cbm = cbm
        self.vbm = vbm
        self.cbm_k_index = cbm_k_index
        self.vbm_k_index = vbm_k_index
        self.cbm_b_index = cbm_b_index
        self.vbm_b_index = vbm_b_index
        self.kpts_array = np.array(kpts_list, dtype=float)
        self.eigens_array_shifted = eigens_array_shifted.reshape(nk, nb)

    def write_kopt(self):
        kpt_cbm_frac = self.kpts_array[self.cbm_k_index]
        kpt_vbm_frac = self.kpts_array[self.vbm_k_index]
        kpt_cbm_cart = kpt_cbm_frac@self.reclat
        kpt_vbm_cart = kpt_vbm_frac@self.reclat

        x_offset = np.linspace(-k_delta, k_delta, 10)
        x_offset = np.column_stack((x_offset, np.zeros(10), np.zeros(10)))
        y_offset = np.linspace(-k_delta, k_delta, 10)
        y_offset = np.column_stack((np.zeros(10), y_offset, np.zeros(10)))
        k_cx_cart = kpt_cbm_cart + x_offset
        k_cy_cart = kpt_cbm_cart + y_offset
        k_vx_cart = kpt_vbm_cart + x_offset
        k_vy_cart = kpt_vbm_cart + y_offset
        # cat 4 10*3 arrays to 40*3 array
        kpts_array_cart = np.concatenate(
            (k_cx_cart, k_cy_cart, k_vx_cart, k_vy_cart), axis=0)
        kpts_array_frac = kpts_array_cart@LA.inv(self.reclat)
        f_emass_opt = open("edge_info/kopt_emass", "w")
        f_emass_opt.write("List KPOINTS\n")
        f_emass_opt.write("40\n")
        f_emass_opt.write("Reciprocal\n")
        for i in range(40):
            f_emass_opt.write("{:.6f} {:.6f} {:.6f} 1\n".format(
                kpts_array_frac[i, 0], kpts_array_frac[i, 1], kpts_array_frac[i, 2]))
        f_emass_opt.close()
        # write kpts for cbm and vbm, for deformation potential constant calculation
        f_edge_opt = open("edge_info/kopt_edge", "w")
        f_edge_opt.write("List KPOINTS\n")
        f_edge_opt.write("2\n")
        f_edge_opt.write("Reciprocal\n")
        f_edge_opt.write("{:.6f} {:.6f} {:.6f} 1\n".format(
            kpt_cbm_frac[0], kpt_cbm_frac[1], kpt_cbm_frac[2]))
        f_edge_opt.write("{:.6f} {:.6f} {:.6f} 1\n".format(
            kpt_vbm_frac[0], kpt_vbm_frac[1], kpt_vbm_frac[2]))
        f_edge_opt.close()

    def get_emass(self):
        # n_seg=10
        cbm_b_index = self.cbm_b_index
        vbm_b_index = self.vbm_b_index
        YY_cbm_x = self.eigens_array_shifted[0:10, cbm_b_index].reshape(10)/27.211386245988
        YY_cbm_y = self.eigens_array_shifted[10:20, cbm_b_index].reshape(10)/27.211386245988
        YY_vbm_x = self.eigens_array_shifted[20:30, vbm_b_index].reshape(10)/27.211386245988
        YY_vbm_y = self.eigens_array_shifted[30:40, vbm_b_index].reshape(10)/27.211386245988
        kpts = self.kpts_array
        XX_cbm_x = LA.norm((kpts[9]-kpts[0]) @ self.reclat) * 2 * np.pi * np.linspace(0, 1, 10) * 0.52917721067
        XX_cbm_y = LA.norm((kpts[19]-kpts[10]) @ self.reclat) * 2 * np.pi * np.linspace(0, 1, 10) * 0.52917721067
        XX_vbm_x = LA.norm((kpts[29]-kpts[20]) @ self.reclat) * 2 * np.pi * np.linspace(0, 1, 10) * 0.52917721067
        XX_vbm_y = LA.norm((kpts[39]-kpts[30]) @ self.reclat) * 2 * np.pi * np.linspace(0, 1, 10) * 0.52917721067
        fit_cbm_x = np.polyfit(XX_cbm_x, YY_cbm_x, 2)
        fit_cbm_y = np.polyfit(XX_cbm_y, YY_cbm_y, 2)
        fit_vbm_x = np.polyfit(XX_vbm_x, YY_vbm_x, 2)
        fit_vbm_y = np.polyfit(XX_vbm_y, YY_vbm_y, 2)
        self.emass_cbm_x = 1**2 / fit_cbm_x[0]
        self.emass_cbm_y = 1**2 / fit_cbm_y[0]
        self.emass_vbm_x = 1**2 / fit_vbm_x[0]
        self.emass_vbm_y = 1**2 / fit_vbm_y[0]
        return self.emass_cbm_x, self.emass_cbm_y, self.emass_vbm_x, self.emass_vbm_y


if __name__ == "__main__":
    os.chdir("result/1_1.000")
    vasp_out = VaspOut()
    vasp_out.get_e_tot()
    vasp_out.get_e_fermi()
    vasp_out.get_vacumm_level_z()
    vasp_out.get_band_structure_Kopt()
    os.makedirs("edge_info", exist_ok=True)
    vasp_out.write_kopt()
    print(vasp_out.e_tot)
    print(vasp_out.e_fermi)
    print(vasp_out.cell)
    # print(vasp_out.kpts_array)
