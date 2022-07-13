# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 23:29:57 2022

@author: Daniel Blatman

# This code is a modification of qiskit code
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
from sklearn.svm import SVC
import numpy as np
from qiskit_machine_learning.kernels import QuantumKernel
from typing import Sequence
from qiskit_machine_learning.utils.loss_functions import KernelLoss

class MySVCLoss(KernelLoss):
    """
    This class replaces the standard class SVCLoss in order to have the
    flexibility to change the hyperparameter C.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments to pass to SVC constructor within
                      SVCLoss evaluation.
        """
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel: QuantumKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        # Bind training parameters
        quantum_kernel.assign_user_parameters(parameter_values)

        # Get estimated kernel matrix
        kmatrix = quantum_kernel.evaluate(np.array(data))

        # Train a quantum support vector classifier
        svc = SVC(kernel="precomputed", **self.kwargs)
        svc.fit(kmatrix, labels)

        # Get dual coefficients
        dual_coefs = svc.dual_coef_[0]

        # Get support vectors
        support_vecs = svc.support_

        # Prune kernel matrix of non-support-vector entries
        kmatrix = kmatrix[support_vecs, :][:, support_vecs]

        # Calculate loss
        loss = np.sum(np.abs(dual_coefs)) - (0.5 * (dual_coefs.T @ kmatrix @ dual_coefs))

        return loss
