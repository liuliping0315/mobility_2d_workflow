import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple
import time

import numpy as np
import pandas as pd

from dflow import (
    RemoteExecutor,
    SlurmRemoteExecutor,
    Step,
    Workflow,
    upload_artifact,
    download_artifact,
    argo_range,
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Slices,
    Artifact,
    upload_packages,
)

from post_process import gen_strain, gen_kpoints, VaspOut
from my_cluster import (
    my_host,    # string, domain or ip
    my_port,    # int
    my_user,    # string, username
)

key_path = os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa")

strain_list = [0.990, 0.995, 1.000, 1.005, 1.010]
strain_str = ["{:1d}_{:.3f}".format(i, j) for j in strain_list for i in [0, 1]]
ratios = [0.990, 0.995, 1.000, 1.005, 1.010]

remote_test = SlurmRemoteExecutor(
    host=my_host,
    port=my_port,
    username=my_user,
    private_key_file=key_path,
    header="#!/bin/bash\n\
    #SBATCH -o job.%j.out\n\
    #SBATCH -p cpu\n\
    #SBATCH -J std_vasp\n\
    #SBATCH --nodes=1\n\
    #SBATCH --ntasks-per-node=64\n\
    source ~/.bashrc\n\
    mamba activate dflow\n\
    pwd\n\
    ",
    # workdir="/data/home/liulp/dflow/workflows/test",
    workdir="/data/home/liulp/dflow/workflows/{{workflow.name}}/{{pod.name}}",
)


class InitData(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "input": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "output": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        os.chdir(op_in["input"])
        print("pwd: ", os.getcwd())
        gen_kpoints()  # create KPOINTS and KPOINTS_OPT

        op_out = OPIO({
            "output": Path(op_in["input"]),
        })
        return op_out


class VASPRun(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "input": Artifact(Path),
            "k_opt_file": str,
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "output": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["input"])
        command = "module load compiler mkl mpi vasp/0.6.4.0; mpirun -n 64 vasp_std | tee out"
        if op_in["k_opt_file"] != "KPOINTS_OPT" and os.path.exists(op_in["k_opt_file"]):
            print("using opt file: ", op_in["k_opt_file"])
            command = "module load compiler mkl mpi vasp/0.6.4.0;\
/usr/bin/cp {} KPOINTS_OPT;\
mpirun -n 64 vasp_std | tee out".format(op_in["k_opt_file"])

        os.system(command)
        os.chdir(cwd)
        op_out = OPIO({
            "output": Path(op_in["input"]),
        })
        return op_out


class PostBandEdge(OP):
    # process new effective mass
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "input": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "output": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        os.chdir(op_in["input"])
        my_vasp_out = VaspOut()
        my_vasp_out.get_cell()
        my_vasp_out.get_e_tot()
        my_vasp_out.get_e_fermi()
        # my_vasp_out.get_vacumm_level_z()
        my_vasp_out.get_band_structure_Kopt()
        np.savetxt("kpts_array", my_vasp_out.kpts_array)
        # cbm and vbm k index, 2 integer
        band_edge_k_index = np.array(
            [my_vasp_out.cbm_k_index, my_vasp_out.vbm_k_index], dtype=int)
        os.makedirs("edge_info", exist_ok=True)
        my_vasp_out.write_kopt()  # create kopt_emass, kopt_edge

        shutil.copy("POSCAR", "edge_info")
        shutil.copy("INCAR", "edge_info")
        shutil.copy("POTCAR", "edge_info")
        shutil.copy("KPOINTS", "edge_info")
        # shutil.copy("kopt_emass", "edge_info")
        # shutil.copy("kopt_edge", "edge_info")
        op_out = OPIO({
            "output": Path(op_in["input"])/"edge_info",
        })
        return op_out


class PostEmass(OP):
    # process new effective mass
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "input": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "emass": List[float],
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        os.chdir(op_in["input"])
        my_vasp_out = VaspOut()
        # these are excuted in __init__
        # my_vasp_out.get_cell()
        # my_vasp_out.get_e_tot()
        # my_vasp_out.get_e_fermi()
        # # my_vasp_out.get_vacumm_level_z()
        my_vasp_out.get_band_structure_Kopt()
        cx, cy, vx, vy = my_vasp_out.get_emass()

        op_out = OPIO({
            "emass": [cx, cy, vx, vy],
            # "output": Path(op_in["input"]),
        })
        return op_out


class ApplyStrain(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "input": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "output": Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        os.chdir(op_in["input"])
        gen_strain()

        output_paths = [Path(op_in["input"]) /
                        p for p in strain_str]

        op_out = OPIO({
            "output": output_paths,
        })
        return op_out


class PostMain(OP):
    # process main results, including:
    #   elastic constant 2D: total energy
    #   deformation potential constant: E_fermi, E_vacuum, band structure
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "input": Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "output": Artifact(Path),
            "c2d": List[float],
            "dfc": List[float],
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cwd1 = os.getcwd()

        print("pp, cwd1: ", cwd1)
        print("pp, op_input[0]:")
        print(op_in["input"][0])

        os.chdir(op_in["input"][0])
        os.chdir("..")
        cwd = os.getcwd()
        outputs = []
        for i in range(10):
            os.chdir(os.path.join(cwd, strain_str[i]))
            tmp_out = VaspOut()
            tmp_out.get_band_structure_Kopt()
            outputs.append(tmp_out)
            os.chdir("..")
        os.chdir(cwd)
        cbm_b_index = outputs[0].cbm_b_index
        vbm_b_index = outputs[0].vbm_b_index
        cell = outputs[0].cell
        # S = |a X b|
        S0 = np.linalg.norm(np.cross(cell[0], cell[1]))
        E_tot_s = np.array([outputs[i].e_tot for i in range(10)])
        E_fermi_s = np.array([outputs[i].e_fermi for i in range(10)])
        E_cbm_s = np.array([outputs[i].eigens_array_shifted[0][cbm_b_index] for i in range(10)])
        E_vbm_s = np.array([outputs[i].eigens_array_shifted[1][cbm_b_index] for i in range(10)])
        p_c2d_x = np.polyfit(np.array(ratios), E_tot_s[0:5], 2)
        p_c2d_y = np.polyfit(np.array(ratios), E_tot_s[5:10], 2)
        c2d_x = p_c2d_x[0] / S0
        c2d_y = p_c2d_y[0] / S0
        p_def_cbm_x = np.polyfit(np.array(ratios), E_cbm_s[0:5], 1)
        p_def_cbm_y = np.polyfit(np.array(ratios), E_cbm_s[5:10], 1)
        def_cbm_x = p_def_cbm_x[0]
        def_cbm_y = p_def_cbm_y[0]
        p_def_vbm_x = np.polyfit(np.array(ratios), E_vbm_s[0:5], 1)
        p_def_vbm_y = np.polyfit(np.array(ratios), E_vbm_s[5:10], 1)
        def_vbm_x = p_def_vbm_x[0]
        def_vbm_y = p_def_vbm_y[0]
        np.savetxt("c2d.txt", np.array([c2d_x, c2d_y]))
        np.savetxt("def_cbm.txt", np.array([def_cbm_x, def_cbm_y]))
        np.savetxt("def_vbm.txt", np.array([def_vbm_x, def_vbm_y]))

        print("pp, cwd: ", cwd)
        op_out = OPIO({
            "output": Path(cwd),
            "c2d": [c2d_x, c2d_y],
            "dfc": [def_cbm_x, def_cbm_y, def_vbm_x, def_vbm_y],
        })
        return op_out


class CollectResult(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "emass": List[float],   # 4, cx, cy, vx, vy
            "c2d": List[float],     # 4, [c2d_x, c2d_y]*2 (that is c11, c22, c11, c22)
            "dfc": List[float],     # 4, cx, cy, vx, vy
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "output": Artifact(Path),
            "emass": List[float],   # 4, cx, cy, vx, vy
            "c2d": List[float],     # 4, [c2d_x, c2d_y]*2 (that is c11, c22, c11, c22)
            "dfc": List[float],     # 4, cx, cy, vx, vy
            "mobility": List[float],  # 4, cx, cy, vx, vy
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        print("pwd: ", os.getcwd())
        print("emass (m_e): ", op_in["emass"])
        print("c2d (eV): ", op_in["c2d"])
        print("dfc (eV): ", op_in["dfc"])
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
        np.savetxt("result/emass.txt", emass, header="m_e_x, m_e_y, m_h_x, m_h_y (m_e)")
        np.savetxt("result/c2d.txt", c2d, header="c11, c22, c11, c22 (Hartree/Bohr**2)")
        np.savetxt("result/dfc.txt", dfc, header="dfc_cbm_x, my_cbm_y, dfc_vbm_x, dfc_vbm_y (Hartree)")
        np.savetxt("result/mu_au.txt", mu_au, header="mu_cbm_x, my_cbm_y, mu_vbm_x, mu_vbm_y (atomic unit)")
        np.savetxt("result/mu_si.txt", mu_si, header="mu_cbm_x, my_cbm_y, mu_vbm_x, mu_vbm_y (m**2/V/s)")
        np.savetxt("result/mu_common.txt", mu_si*10000, header="mu_cbm_x, my_cbm_y, mu_vbm_x, mu_vbm_y (cm**2/V/s)")
        # combine all the results to a pandas dataframe, then save to csv
        labels = np.array(["electron-X", "electron-Y", "hole-X", "hole-Y"], dtype=str)
        df = pd.DataFrame(
            {"labels": labels, "emass-au": emass, "c2d-au": c2d, "dfc-au": dfc, "mobility-au": mu_au, "mobility-si": mu_si, "mobility-common": mu_si*10000},
        )

        df.to_csv("result/result.csv", index=False, header=True)

        op_out = OPIO({
            "output": Path("result"),
            "emass": emass.tolist(),
            "c2d": c2d.tolist(),
            "dfc": dfc.tolist(),
            "mobility": mu_si.tolist(),
        })
        return op_out


def test():
    wf.add(init_data)           # create spglib KPOINTS_OPT
    wf.add(vasp_band_edge)      # run vasp to get band structure
    wf.add(post_band_edge)      # create kopt_emass, kopt_edge
    wf.add(vasp_emass)          # run vasp with KPOINTS_OPT=kopt_emass
    wf.add(post_emass)          # calculate emass, outputs = [cbm_x, cbm_y, vbm_x, vbm_y]
    wf.add(apply_strain)        # apply strain
    wf.add(vasp_main)           # run slices in strained directory
    wf.add(post_main)           # collect c2d, deformation potential constant
    wf.add(collect_result)      # collect results


if __name__ == "__main__":
    upload_packages.append("post_process.py")
    init_data = Step("Init-Data",
                     PythonOPTemplate(InitData),
                     artifacts={"input": upload_artifact(
                         ["mobility"])},
                     executor=remote_test,
                     )

    vasp_band_edge = Step("VASP-Band-Edge",
                          PythonOPTemplate(VASPRun),
                          artifacts={
                              "input": init_data.outputs.artifacts["output"]},
                          parameters={"k_opt_file": "KPOINTS_OPT"},
                          key="vasp-band-edge",
                          executor=remote_test,
                          )
    post_band_edge = Step("Post-Band-Edge",
                          PythonOPTemplate(PostBandEdge),
                          artifacts={
                              "input": vasp_band_edge.outputs.artifacts["output"]},
                          executor=remote_test,
                          )
    vasp_emass = Step("VASP-Emass",
                      PythonOPTemplate(VASPRun),
                      artifacts={
                          "input": post_band_edge.outputs.artifacts["output"]},
                      parameters={"k_opt_file": "kopt_emass"},
                      key="vasp-emass",
                      executor=remote_test,
                      )
    post_emass = Step("Post-Emass",
                      PythonOPTemplate(PostEmass),
                      artifacts={
                          "input": vasp_emass.outputs.artifacts["output"]},
                      executor=remote_test,
                      )
    apply_strain = Step("Apply-Strain",
                        PythonOPTemplate(ApplyStrain),
                        artifacts={
                            "input": post_band_edge.outputs.artifacts["output"]},
                        executor=remote_test,
                        )

    vasp_main = Step("VASP-Main",
                     PythonOPTemplate(VASPRun,
                                      slices=Slices("{{item}}",
                                                    input_artifact=["input"],
                                                    output_artifact=["output"],
                                                    ),
                                      ),
                     artifacts={
                         "input": apply_strain.outputs.artifacts["output"]},
                     parameters={"k_opt_file": ["kopt_edge"]*10},
                     with_param=argo_range(10),
                     key="vaspmain-{{item}}",
                     executor=remote_test,
                     )
    post_main = Step("Post-Main",
                     PythonOPTemplate(PostMain),
                     artifacts={
                         "input": vasp_main.outputs.artifacts["output"]},
                     executor=remote_test,
                     )
    collect_result = Step("Collect",
                          PythonOPTemplate(CollectResult),
                          parameters={
                              "emass": post_emass.outputs.parameters["emass"],
                              "c2d": post_main.outputs.parameters["c2d"],
                              "dfc": post_main.outputs.parameters["dfc"],
                          },
                          executor=remote_test,
                          )

    wf = Workflow("mobility")
    wf.add(init_data)                       # create spglib KPOINTS_OPT
    wf.add(vasp_band_edge)                  # run vasp to get band structure
    wf.add(post_band_edge)                  # create kopt_emass, kopt_edge
    wf.add([vasp_emass, apply_strain])      # run vasp with KPOINTS_OPT=kopt_emass; # apply strain
    wf.add([post_emass, vasp_main])         # calculate emass, outputs = [cbm_x, cbm_y, vbm_x, vbm_y]; # run slices in strained directory

    wf.add(post_main)           # collect c2d, deformation potential constant
    wf.add(collect_result)      # collect results
    wf.submit()

    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(10)
    assert (wf.query_status() == "Succeeded")
    steps = wf.query_step(name="Collect")
    print("len of steps:")
    print(len(steps))
    assert (steps[0].phase == "Succeeded")
    print("result a.u.:")
    print("emass: ", steps[0].outputs.parameters["emass"])
    print("c2d: ", steps[0].outputs.parameters["c2d"])
    print("dfc: ", steps[0].outputs.parameters["dfc"])
    print("mobility: ", steps[0].outputs.parameters["mobility"])
    download_artifact(
        steps[0].outputs.artifacts["output"],
        path=r"result",
    )
