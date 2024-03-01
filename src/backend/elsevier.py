from typing import Tuple

import numpy as np
import scipy.sparse as sparse


def get_data(year: int) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    raise NotImplementedError("Not implemented in the Elsevier backend.")


def update_posterior(auids: np.ndarray, posterior: np.ndarray, year: int) -> None:
    raise NotImplementedError("Not implemented in the Elsevier backend.")
