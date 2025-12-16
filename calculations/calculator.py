import torch
import numpy as np
from mace.calculators.mace import MACECalculator
from typing import List, Union
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator, all_changes
from e3nn import o3
from mace.modules.utils import extract_invariant

class ModifiedMACECalculator(MACECalculator):
    def __init__(
        self,
        model_paths: Union[list, str, None] = None,
        models: Union[List[torch.nn.Module], torch.nn.Module, None] = None,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        info_keys=None,
        arrays_keys=None,
        model_type="MACE",
        compile_mode=None,
        fullgraph=True,
        enable_cueq=False,
        **kwargs,
    ):
        super().__init__(
            model_paths=model_paths,
            models=models,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key=charges_key,
            info_keys=info_keys,
            arrays_keys=arrays_keys,
            model_type=model_type,
            compile_mode=compile_mode,
            fullgraph=fullgraph,
            enable_cueq=enable_cueq,
            **kwargs,
        )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = self._clone_batch(batch_base)
            node_heads = batch["head"][batch["batch"]]
            num_atoms_arange = torch.arange(batch["positions"].shape[0])
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not self.use_compile
        else:
            compute_stress = False

        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_models, len(atoms)
        )

        num_interactions = int(self.models[0].num_interactions)
        irreps_out = o3.Irreps(str(self.models[0].products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
        per_layer_features[-1] = (
            num_invariant_features  # Equivariant features not created for the last layer
        )
        to_keep = np.sum(per_layer_features[:num_interactions])

        self.results = {}
        self.results['batch'] = []
        self.results['outs'] = []
        self.results['descriptors'] = []
        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            self.results['batch'].append(batch)
            out = model(
                batch.to_dict(),
                compute_stress=compute_stress,
                training=True,
                compute_edge_forces=self.compute_atomic_stresses,
                compute_atomic_stresses=self.compute_atomic_stresses,
            )
            self.results['outs'].append(batch)
            self.results['descriptors'].append(out["node_feats"])
            if self.model_type in ["MACE", "EnergyDipoleMACE"]:
                ret_tensors["energies"][i] = out["energy"].detach()
                ret_tensors["node_energy"][i] = (out["node_energy"] - node_e0).detach()
                ret_tensors["forces"][i] = out["forces"].detach()
                if out["stress"] is not None:
                    ret_tensors["stress"][i] = out["stress"].detach()
            if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
                ret_tensors["dipole"][i] = out["dipole"].detach()
            if self.model_type in ["MACE"]:
                if out["atomic_stresses"] is not None:
                    ret_tensors.setdefault("atomic_stresses", []).append(
                        out["atomic_stresses"].detach()
                    )
                if out["atomic_virials"] is not None:
                    ret_tensors.setdefault("atomic_virials", []).append(
                        out["atomic_virials"].detach()
                    )

        self.results['descriptors'] = [
            extract_invariant(
                descriptor,
                num_layers=num_interactions,
                num_features=num_invariant_features,
                l_max=l_max,
            )
            for descriptor in self.results['descriptors']
        ]
        self.results['descriptors'] = [
            descriptor[:, :to_keep] for descriptor in self.results['descriptors']
        ]

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            self.results["energy"] = (
                    torch.mean(ret_tensors["energies"], dim=0).cpu().item()
                    * self.energy_units_to_eV
            )
            self.results["free_energy"] = self.results["energy"]
            self.results["node_energy"] = (
                torch.mean(ret_tensors["node_energy"], dim=0).cpu().numpy()
            )
            self.results["forces"] = (
                    torch.mean(ret_tensors["forces"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A
            )
            if self.num_models > 1:
                self.results["energies"] = (
                        ret_tensors["energies"].cpu().numpy() * self.energy_units_to_eV
                )
                self.results["energy_var"] = (
                        torch.var(ret_tensors["energies"], dim=0, unbiased=False)
                        .cpu()
                        .item()
                        * self.energy_units_to_eV
                )
                self.results["forces_comm"] = (
                        ret_tensors["forces"].cpu().numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A
                )
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A ** 3
                )
                if self.num_models > 1:
                    self.results["stress_var"] = full_3x3_to_voigt_6_stress(
                        torch.var(ret_tensors["stress"], dim=0, unbiased=False)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A ** 3
                    )
            if "atomic_stresses" in ret_tensors:
                self.results["stresses"] = (
                        torch.mean(torch.stack(ret_tensors["atomic_stresses"]), dim=0)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A ** 3
                )
            if "atomic_virials" in ret_tensors:
                self.results["virials"] = (
                        torch.mean(torch.stack(ret_tensors["atomic_virials"]), dim=0)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=0).cpu().numpy()
            )
            if self.num_models > 1:
                self.results["dipole_var"] = (
                    torch.var(ret_tensors["dipole"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )


