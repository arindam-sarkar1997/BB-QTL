"""Check the ID join and complete case rule used before GP fitting."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from yeast_inputs import load_drug_inputs


class DrugInputTests(unittest.TestCase):
    def test_genotypes_follow_ids_and_require_all_four_fitness_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            geno_path = root / "geno.npy"
            ids_path = root / "ids.tsv"
            snp_info_path = root / "snp.tsv"
            pheno_path = root / "fitness"
            np.save(geno_path, np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8))
            np.savetxt(ids_path, [30, 10, 20], fmt="%d")
            pd.DataFrame({"Index": [0, 1]}).to_csv(snp_info_path, sep="\t", index=False)

            for dose in ("CON", "QMIC", "HMIC", "FMIC"):
                folder = pheno_path / "FLU" / dose
                folder.mkdir(parents=True)
                rows = pd.DataFrame({
                    "ID": [10, 20, 30],
                    "s": [1.0, 2.0, 3.0],
                    "stderr(s)": [0.1, 0.2, 0.3],
                })
                if dose == "QMIC":
                    rows.loc[rows.ID == 20, "s"] = np.nan
                if dose == "FMIC":
                    rows["s"] += 10
                rows.to_csv(folder / "fitness_ALLreps_with_ids.tsv", sep="\t", index=False)

            geno, y, y_var, ids = load_drug_inputs(
                "FLU", "FMIC", geno_path, ids_path, snp_info_path, pheno_path
            )
            np.testing.assert_array_equal(ids, [30, 10])
            np.testing.assert_array_equal(geno, [[1, 0], [0, 1]])
            np.testing.assert_allclose(y, [13, 11])
            np.testing.assert_allclose(y_var, [0.09, 0.01], rtol=1e-6)

    def test_dose_must_belong_to_drug(self):
        with self.assertRaisesRegex(ValueError, "not available"):
            load_drug_inputs("PUL", "QMIC")


if __name__ == "__main__":
    unittest.main()
