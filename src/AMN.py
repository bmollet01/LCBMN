"""

This script includes code from 'Artificial Metabolic Models' by BioRetroSynth Group
available at https://github.com/brsynth/amn_release

Original code is licensed under the MIT License.
See LICENSE or third_party/some-library/LICENSE for details.
"""

###############################################################################
# This library create training sets for AMN
# Trainig sets are either based on experimental datasets
# or FBA (Cobrapy) simulations
# Authors: Jean-loup Faulon jfaulon@gmail.com and Bastien Mollet
###############################################################################

from __future__ import print_function
import os
import sys
import math
import numpy as np
import pandas
from sklearn.utils import shuffle
import os
import keras
import keras.backend as K
import copy
import tensorflow as tf
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Lambda
from keras.layers import concatenate
from keras.utils.generic_utils import get_custom_objects
from keras.utils.generic_utils import CustomObjectScope
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

tf.config.set_visible_devices([], "GPU")
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != "GPU"
try:
    import cobra
    # import cobra.test # was crashing the colab implementation
    import cobra.manipulation as manip
except ImportError:
    print("Cobra is not installed. Skipping that part of the code.")

sys.setrecursionlimit(10000)  # for row_echelon function

###############################################################################
# IOs with pandas
###############################################################################


def read_csv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    if filename[-3:] != "csv":
        filename += ".csv"
    dataframe = pandas.read_csv(filename, header=0)
    dataframe = dataframe.dropna()
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:, :])
    return HEADER, DATA


def MinMaxScaler(data, Min_Scaler=-1.0e12, Max_Scaler=1.0e12):
    # MinMax standardize np array data
    if Max_Scaler == 1.0e12:  # Scale
        Min_Scaler, Max_Scaler = np.min(data), np.max(data)
        data = (data - Min_Scaler) / (Max_Scaler - Min_Scaler)
    else:  # Descale
        data = data * (Max_Scaler - Min_Scaler) + Min_Scaler
        Min_Scaler = -1.0e12
        Max_Scaler = 1.0e12
    return data, Min_Scaler, Max_Scaler


def MaxScaler(data, Max_Scaler=1.0e12):
    # Max standardize np array data
    if Max_Scaler == 1.0e12:  # Scale
        Max_Scaler = np.max(data)
        data = data / Max_Scaler
    else:  # Descale
        data = data * Max_Scaler
        Max_Scaler = 1.0e12
    return data, Max_Scaler


def read_XY(filename, nY=1, scaling=""):
    # Format data for training
    # Function read_training_data is defined in module (1)
    # if scaling == 'X' X is scaled
    # if scaling == 'Y' Y is scaled
    # if scaling == 'XY' X and Y are scaled
    _, XY = read_csv(filename)

    XY = np.asarray(XY)
    X = XY[:, :-nY]
    Y = XY[:, -nY:]
    X, _, _ = MinMaxScaler(X) if scaling == "X" or scaling == "XY" else X, 0, 0
    Y, _, _ = MinMaxScaler(Y) if scaling == "Y" or scaling == "XY" else Y, 0, 0
    return X, Y


###############################################################################
# Cobra's model utilities and matrices (written by Bastien Mollet)
###############################################################################

# Cobra utilities and stoichiometric derived matrices
def get_index_from_id(name, L):
    # Return index in L of id name
    for i in range(len(L)):
        if L[i].id == name:
            return i
    return -1


def get_objective(model):
    # Get the reaction carring the objective
    # Someone please tell me if there is
    # a clearner way in Cobra to get
    # the objective reaction

    r = str(model.objective.expression)
    r = r.split()
    r = r[0].split("*")
    obj_id = r[1]

    # line below crash if does not exist
    r = model.reactions.get_by_id(obj_id)

    return obj_id


def get_matrices(model, medium, measure, reactions):
    # Get matrices for AMN_QP and AMN_Wt
    # Return
    # - S [mxn]: stochiometric matrix
    # - V2M [mxn]: to compute metabolite
    #        production fluxes from reaction fluxes
    # - M2V [mxn]: to compute reaction fluxes
    #        from substrate production fluxes
    # - Pin [n_in x n]: to go from reactions to medium fluxes
    # - Pout [n_out x n]: to go from reactions to measured fluxes

    # m = metabolite, n = reaction/v/flux, p = medium
    S = np.asarray(cobra.util.array.create_stoichiometric_matrix(model))
    n, m, n_in, n_out = S.shape[1], S.shape[0], len(medium), len(measure)

    # Get V2M and M2V from S
    V2M, M2V = S.copy(), S.copy()
    for i in range(m):
        for j in range(n):
            if S[i][j] < 0:
                V2M[i][j] = 0
                M2V[i][j] = -1 / S[i][j]
            else:
                V2M[i][j] = S[i][j]
                M2V[i][j] = 0
    M2V = np.transpose(M2V)

    # Boundary matrices from reaction to medium fluxes
    Pin, i = np.zeros((n_in, n)), 0
    for rid in medium:
        j = get_index_from_id(rid, reactions)
        Pin[i][j] = 1
        i = i + 1

    # Experimental measurements matrix from reaction to measured fluxes
    Pout, i = np.zeros((n_out, n)), 0
    for rid in measure:
        j = get_index_from_id(rid, reactions)
        Pout[i][j] = 1
        i = i + 1

    return S, Pin, Pout, V2M, M2V


def row_echelon(A, C):
    # Return Row Echelon Form of matrix A and the matrix C
    # will be used to perform all the operations on b later
    # This function is recursive, it works by turning the first
    # non-zero row to 1. Then substract all the other row
    # to turn them to 0. Thus, perform the same operation on
    # the second row/ second column.
    # If matrix A has no columns or rows, it is already in REF,
    # so we return itself, it's the end of the recursion.

    r, c = A.shape
    if r == 0 or c == 0:
        return A, C

    # We search for non-zero element in the first column.
    # (If/else is used in a strange wy but the Else is skipped
    # if break happens in if)
    # ( Else can't be used in the for)
    for i in range(len(A)):
        if A[i, 0] != 0:
            break
    else:
        # If all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:, 1:], C)
        # and then add the first zero-column back
        return np.hstack([A[:, :1], B[0]]), C

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        C_ith_row = C[i].copy()
        A[i] = A[0]
        C[i] = C[0]
        C[0] = C_ith_row
        A[0] = ith_row

    # We divide first row by first element in it
    # Here it's important to first change C as the value
    Scaling_factor = A[
        0, 0
    ]  # Keep this value in memory in case it makes too high values.
    C[0] = C[0] / Scaling_factor
    A[0] = A[0] / Scaling_factor

    # We subtract all subsequent rows with first row
    # (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    C[1:] -= C[0] * A[1:, 0:1]
    A[1:] -= A[0] * A[1:, 0:1]

    # Controling values to remain differentiable ####
    up_bound = np.amax(A[1:], 1)
    for i in range(1, len(up_bound)):
        max_row = up_bound[i - 1]
        if max_row >= 1000:
            C[i] = C[i] / max_row
            A[i] = A[i] / max_row

    # If the scaling factor is too small, values in A[0] can be too high
    if np.amax(A[0]) >= 1000:
        C[0] = C[0] * Scaling_factor
        A[0] = A[0] * Scaling_factor
    # End of the controling part ####

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:, 1:], C[1:, :])

    # we add first row and first (zero) column, and return
    subA = np.vstack([A[:1], np.hstack([A[1:, :1], B[0]])])
    subC = np.vstack([C[:1], B[1]])
    return subA, subC


def get_B(model, S, medium, verbose=False):
    # A matrix used to get boundary vectors in get_matrices_LP
    _, m, p = S.shape[1], S.shape[0], len(medium)
    B, i = np.zeros((p, m)), 0
    # print(p)
    for rid in medium:
        k = get_index_from_id(rid, model.reactions)
        r = model.reactions[k]
        # print(r.products)
        p = r.products[0]  # medium reactions have only one product
        j = get_index_from_id(p.id, model.metabolites)
        B[i][j] = 1
        i = i + 1
    if verbose:
        print("When you get B: ", B[0], B.shape)
    # print("Where is B non-zero: ", np.nonzero(B))
    return B


def get_matrices_LP(model, mediumbound, X, S, Pin, medium, obj, verbose=False):
    # Get matrices and vectors for LP cells from
    # Y. Yang et al. Mathematics & Computers in Simulation 101, 103–112, (2014)
    # Outputs:
    # - Sint [mxn] [m metabolites, n fluxes]
    #   For EB the stoichiometric matrix S where columns corresponding
    #   to intake fluxes are zeroed out
    #   For UB same as EB + rows corresponding to metabolites entries
    #   are zeroed out
    # - Sext [mxn]
    #   For EB = I [nxn] (m=n)
    #   For UB the stoichiometric matrix S where only rows
    #   corresponding to internal metabolites are kept + I -
    #   stoichiometric matrix S where only rows
    #   corresponding to internal metabolites are kept
    # - Q = S_int^T (S_int S_int^T)-1 [n x m]
    # - P = Q S_int - I [n x n]
    # - b_int [m]
    #   For EB the extact bound values
    #   For UB = 0
    # - b_ext [m]
    #   For EB = 0
    #   For UB the upper bound values
    # columns in Sb corresponding to medium are zeroed out

    Sb = -np.transpose(S.copy())
    S_int = Sb.copy()

    c = np.zeros(S.shape[1])
    for i in range(len(obj)):
        c[get_index_from_id(obj[i], model.reactions)] = -1.0
        # Here this parameter can be tuned to increase the focus on max c
    c = np.float32(c)

    inputs = np.float32(X)

    if inputs.shape[1] == S.shape[1]:  # Special case inputs = Vsol * noise
        V_all = inputs
    else:  # V =  Pin inputs
        Pin = np.float32(Pin)
        V_all = np.matmul(inputs, Pin)
        # V_all = V_all.numpy()

    for rid in medium:
        i = get_index_from_id(rid, model.reactions)
        Sb[i] = np.zeros(S.shape[0])

    if mediumbound == "UB":
        # print('We are in UB')
        S_ext = Sb.copy()
        Id = -np.identity(S_int.shape[0])
        S_ext_p = -np.copy(S_ext)
        # This part of S_ext ensure that every flux is positive.
        S_ext = np.concatenate((S_ext, Id), axis=1)
        S_ext = np.concatenate((S_ext, S_ext_p), axis=1)
    else:
        # print('We are in EB')
        S_int = Sb.copy()
        S_ext = -np.identity(S_int.shape[0])

    # Triangulate matrix S_int and record row permutation in Transform
    S_int = np.transpose(S_int)
    S_int, Transform = row_echelon(S_int, np.identity(S_int.shape[0]))
    S_int = S_int[~np.all(S_int == 0, axis=1)]  # remove zero line

    # print("transform:", Transform.shape)
    # P and Q
    Q = np.dot(S_int, np.transpose(S_int))
    Q = np.linalg.inv(Q)  # inverse matrix
    Q = np.dot(np.transpose(S_int), Q)

    P = np.dot(Q, S_int)
    P = P - np.identity(P.shape[0])  # -(I-P)

    # b_int and b_ext
    B = get_B(model, S, medium, verbose=verbose)
    b = np.matmul(inputs, B)
    b = np.float32(b)

    if mediumbound == "UB":
        b_int = np.zeros(S_int.shape[0])  # null vector
        # b_int[np.where(b_int==0)] = DEFAULT_UB # breaks the method
        b_int = np.float32(b_int)
        b_ext_all = b
        # This part aims to build the b vector that can be used with 2014.
        # It takes the same input as 2006 but it needs
        # to be added parts with 0 to ensure the different inequalities.
        # As explained for M, b_ext in the UB case ensure 3 constraints.
        # The first one (upper bounds) is set by b_ext.
        # b_add aims to ensure the next two ones.
        new_b_ext = []
        for i in range(len(V_all)):
            V = V_all[i]
            b_ext = b_ext_all[i]
            if verbose:
                print("b_ext before b_add ", b_ext.shape)
            b_add = np.zeros(V.shape[0] + b_ext.shape[0])
            if "ATPM" in model.reactions:
                # ATPM is the only reaction (to our knowledge)
                # with a lower bound.
                # It could be a good update to search for non-zero
                # lower bounds automatically.
                indice = get_index_from_id("ATPM", model.reactions)
                ATPM_LB = model.reactions.get_by_id("ATPM").lower_bound
                b_add[indice] = -ATPM_LB
            # print(b_add)
            b_add = np.transpose(b_add)
            b_ext = np.transpose(b_ext)
            b_used = np.concatenate([b_ext, b_add], axis=0)
            if verbose:
                print("b_ext after b_add ", b_used)
            new_b_ext.append(b_used)
        b_ext = np.array(new_b_ext, dtype=np.float32)

    else:  # EB b_int must be transformed because S_int was passed in row form
        b_int = np.matmul(np.float32(Transform), b.T)
        b_int = np.transpose(b_int[: S_int.shape[0]])
        b_ext = np.zeros(S.shape[1])  # null vector
        # b_ext[np.where(b_ext==0)] = DEFAULT_UB # breaks the method
        b_ext = np.float32(b_ext)

    Sb = np.float32(Sb)
    S_int = np.float32(S_int)
    S_ext = np.float32(S_ext)
    Q = np.float32(Q)
    P = np.float32(P)
    return S_int, S_ext, Q, P, b_int, b_ext, Sb, c


def reduce_model(model, medium, measure, flux, verbose=False):
    # Remove all reactions not in medium having a zero flux
    # Input: the model, the medium, the flux vector (a 2D array)
    # Output: the reduce model

    # Collect reaction to be removed
    remove = {}
    for i in range(flux.shape[1]):
        if (
            np.count_nonzero(flux[:, i]) == 0
            and model.reactions[i].id not in medium
            and model.reactions[i].id not in measure
        ):
            remove[i] = model.reactions[i]

    # Actual deletion
    model.remove_reactions(list(remove.values()))
    manip.delete.prune_unused_reactions(model)
    for m in model.metabolites:
        if len(m.reactions) == 0:
            model.remove_metabolites(m)
    manip.delete.prune_unused_metabolites(model)
    print(
        "reduced numbers of metabolites and reactions:",
        len(model.metabolites),
        len(model.reactions),
    )

    return model


###############################################################################
# Running Cobra
###############################################################################


def run_cobra(
    model,
    objective,
    IN,
    method="FBA",
    verbose=False,
    objective_fraction=0.75,
    cobra_min_flux=1.0e-8,
):
    # Inputs:
    # - model
    # - objective: a list of reactions (first two only are considered)
    # - IN: Initial values for all reaction fluxes
    # - method: FBA or pFBA
    # run FBA optimization to compute recation fluxes on the provided model
    # set the medium using values in dictionary IN.
    # When 2 objectives are given one first maximize the first objective (obj1)
    # then one set the upper and lower bounds for that objective to
    # objective_fraction * obj1 (e.g. objective_fraction = 0.75) and maximize
    # for the second objective
    # Outputs:
    # - FLUX, the reaction fluxes compyted by FBA for all reactions
    # - The value for the objective

    # set the medium and objective

    medium = model.medium  # This is the model medium
    medini = medium.copy()
    for k in medium.keys():  # Reset the medium
        medium[k] = 0
    for k in IN.keys():  # Additional cmpds added to medium
        if k in medium.keys():
            medium[k] = float(IN[k])
    model.medium = medium

    # run FBA for primal objective
    model.objective = objective[0]
    if method == "pFBA":
        solution = cobra.flux_analysis.pfba(model)
    else:
        solution = model.optimize()
    solution_val = solution.fluxes[objective[0]]
    if verbose:
        print("primal objectif =", objective, method, solution_val)

    # run FBA for second objective
    # primal objectif is set to a fraction of its value
    if len(objective) > 1:
        obj = model.reactions.get_by_id(objective[0])
        obj_lb, obj_ub = obj.lower_bound, obj.upper_bound
        obj.lower_bound = objective_fraction * solution_val
        obj.upper_bound = objective_fraction * solution_val
        model.objective = objective[1]
        if method == "pFBA":
            solution = cobra.flux_analysis.pfba(model)
        else:
            solution = model.optimize()
        solution_val = solution.fluxes[objective[1]]
        if verbose:
            print("second objectif =", objective, method, solution_val)

        # reset bounds and objective to intial values
        obj.lower_bound, obj.upper_bound = obj_lb, obj_ub
        model.objective = objective[0]

    # get the fluxes for all model reactions
    FLUX = IN.copy()
    for x in model.reactions:
        if x.id in FLUX.keys():
            FLUX[x.id] = solution.fluxes[x.id]
            if math.fabs(float(FLUX[x.id])) < cobra_min_flux:  # !!!
                FLUX[x.id] = 0

    # Reset medium
    model.medium = medini

    return FLUX, solution_val


###############################################################################
# Generating random medium runing Cobra
###############################################################################


def create_random_medium_cobra(
    model,
    objective,
    medium,
    in_varmed,
    levmed,
    valmed,
    ratmed,
    method="FBA",
    verbose=False,
    cobra_min_objective=1.0e-3,
):
    # Generate a random input and get Cobra's output
    # Input:
    # - model
    # - objective: the reaction fluxes to optimize
    # - medium: list of reaction fluxes in medium
    # - in_varmed: the medium reaction fluxes allowed to change
    #              (can be empty then varmed are drawn at random)
    # - levmed: teh number of level a flux can take
    # - valmed: the maximum value the flux can take
    # - ratmed: the ration of fluxes turned on
    # - method: the method used by Cobra
    # Make sure the medium does not kill the objective
    # i.e. objective > cobra_min_objective
    # Ouput:
    # - Intial reaction fluxes set to medium values

    MAX_iteration = 5  # max numbrer of Cobra's failaure allowed

    medini = model.medium.copy()
    INFLUX = {}
    for r in model.reactions:
        INFLUX[r.id] = 0

    # X = actual number of variable medium turned ON
    L_in_varmed = len(in_varmed)
    if L_in_varmed > 0:
        X = len(in_varmed)
    else:
        X = sum(map(lambda x: x > 1, levmed))  # total number of variable med
        X = np.random.binomial(X, ratmed, 1)[0] if ratmed < 1 else int(ratmed)
        X = 1 if X == 0 else X

    # Indices for minmed varmed
    minmed, varmed = [], []
    for i in range(len(medium)):
        if levmed[i] <= 1:  # mimimum medium indices
            minmed.append(i)
        else:
            if len(in_varmed) > 0:
                if medium[i] not in in_varmed:
                    continue
            varmed.append(i)  # variable medium indices

    # modmed = minmed + varmed if mediumbound == "EB" else varmed

    for iteration in range(MAX_iteration):
        # create random medium choosing X fluxes in varmed at random
        INFLUX = {k: 0 for k in INFLUX.keys()}  # reset
        model.medium = medini  # reset
        varmed = shuffle(varmed)  # that's where random choice occur
        for i in range(len(minmed)):
            j = minmed[i]
            k = medium[j]
            INFLUX[k], model.medium[k] = valmed[j], valmed[j]
        for i in range(X):
            j = varmed[i]
            k = medium[j]
            v = (
                (L_in_varmed + 1)
                * np.random.randint(1, high=levmed[j])
                * valmed[j]
                / (levmed[j] - 1)
            )
            INFLUX[k], model.medium[k] = v, v

        # check with cobra
        try:
            _, obj = run_cobra(model, objective, INFLUX, method=method, verbose=False)
        except ValueError:
            print("Cobra cannot be run start again")
        # except:
        #     print("Cobra cannot be run start again")
        #     treshold, iteration, up, valmed = init_constrained_objective(
        #         objective_value, in_treshold, modmed, valmed, verbose=verbose
        #     )
        #     continue

        if obj < cobra_min_objective:
            continue  # must have some objective

        # We have a solution
        if verbose:
            p = [medium[varmed[i]] for i in range(X)]
            print("pass (varmed, obj):", p, obj)
        break

    model.medium = medini  # reset medium

    return INFLUX


def get_io_cobra(
    model,
    objective,
    medium,
    mediumbound,
    varmed,
    levmed,
    valmed,
    ratmed,
    E,
    method="FBA",
    inf={},
    verbose=False,
):
    # Generate a random input and get Cobra's output
    # Input:
    # - model: the cobra model
    # - objective: the list of objectiev fluxes to maximize
    # - medium: list of reaction fluxes in medium
    # - varmed: the medium reaction fluxes allowed to change
    #            (can be empty then varmed are drawn at random)
    # - levmed: the number of level an uptake flux can take
    # - valmed: the maximum value the flux can take
    # - ratmed: the ration of fluxes turned on
    # - method: the method used by Cobra
    # Output:
    # - X=medium , Y=fluxes for reactions in E

    if inf == {}:
        inf = create_random_medium_cobra(
            model,
            objective,
            medium,
            mediumbound,
            varmed,
            levmed,
            valmed.copy(),
            ratmed,
            method=method,
            verbose=verbose,
        )
    out, obj = run_cobra(model, objective, inf, method=method, verbose=verbose)
    Y = np.asarray(list(out.values()))
    X = np.asarray([inf[medium[i]] for i in range(len(medium))])

    return X, Y


###############################################################################
# Creating, saving and loading training set object
# Training set object used in all modules
###############################################################################


class TrainingSet:
    # All element necessary to run AMN
    # cf. save for definition of parameters
    def __init__(
        self,
        cobraname="",
        mediumname="",
        mediumbound="EB",
        mediumsize=-1,
        objective=[],
        method="FBA",
        measure=[],
        verbose=False,
    ):

        if cobraname == "":
            return  # create an empty object
        if not os.path.isfile(cobraname + ".xml"):
            print(cobraname)
            sys.exit("xml cobra file not found")
        if not os.path.isfile(mediumname + ".csv"):
            print(mediumname)
            sys.exit("medium or experimental file not found")
        self.cobraname = cobraname  # model cobra file
        self.mediumname = mediumname  # medium file
        self.mediumbound = mediumbound  # EB or UB
        self.method = method
        self.model = cobra.io.read_sbml_model(cobraname + ".xml")
        self.reduce = False
        self.allmatrices = True

        # set medium
        H, M = read_csv(mediumname)
        if "EXP" in self.method:  # Reading X, Y
            if mediumsize < 1:
                sys.exit("must indicate medium size with experimental dataset")
            # 3 lines below added Nov 2, 2023 by JLF
            medium = [H[i] for i in range(mediumsize)]
            levmed = [len(set(M[:, i])) for i in range(mediumsize)]
            valmed = [np.mean(M[:, i]) for i in range(mediumsize)]
            self.medium = medium
            self.levmed, self.valmed, self.ratmed = levmed, valmed, 0
            self.X = M[:, 0: len(medium)]
            self.Y = M[:, len(medium):]
            self.size = self.Y.shape[0]
        else:
            self.medium = H[1:]
            self.levmed = [float(i) for i in M[0, 1:]]
            self.valmed = [float(i) for i in M[1, 1:]]
            self.ratmed = float(M[2, 1])
            (self.X, self.Y) = (
                np.asarray([]).reshape(0, 0),
                np.asarray([]).reshape(0, 0),
            )

        if verbose:
            print("medium:", self.medium)
            print("levmed:", self.levmed)
            print("valmed:", self.valmed)
            print("ratmed:", self.ratmed)

        # set objectve and measured reactions lists
        self.objective = [get_objective(self.model)] \
        if objective == [] else objective
        self.measure = [r.id for r in self.model.reactions] if measure == [] else measure
        if verbose:
            print("objective: ", self.objective)
            print("measurements size: ", len(self.measure))

        # compute matrices and objective vector for AMN
        self.S, self.Pin, self.Pout, self.V2M, self.M2V = get_matrices(
            self.model, self.medium, self.measure, self.model.reactions
        )

    def reduce_and_run(self, verbose=False):
        # reduce a model recompute matrices and rerun cobra
        # with the provided training set
        if len(self.measure) == len(self.model.reactions):
            measure = []
        else:
            self.measure = measure
        self.model = reduce_model(
            self.model, self.medium, measure, self.Y, verbose=verbose
        )
        self.measure = [r.id for r in self.model.reactions] if measure == [] else measure
        self.get(sample_size=self.size, reduce=True, verbose=verbose)

    def save(self, filename, reduce=False, verbose=False):
        # save cobra model in xml and parameter in npz (compressed npy)
        self.reduce = reduce
        if self.reduce:
            self.reduce_and_run(verbose=verbose)
        # Recompute matrices
        self.S, self.Pin, self.Pout, self.V2M, self.M2V = get_matrices(
            self.model, self.medium, self.measure, self.model.reactions
        )
        (
            self.S_int,
            self.S_ext,
            self.Q,
            self.P,
            self.b_int,
            self.b_ext,
            self.Sb,
            self.c,
        ) = get_matrices_LP(
            self.model,
            self.mediumbound,
            self.X,
            self.S,
            self.Pin,
            self.medium,
            self.objective,
        )
        # save cobra file
        cobra.io.write_sbml_model(self.model, filename + ".xml")
        # save parameters
        np.savez_compressed(
            filename,
            cobraname=filename,
            reduce=self.reduce,
            mediumname=self.mediumname,
            mediumbound=self.mediumbound,
            objective=self.objective,
            method=self.method,
            size=self.size,
            medium=self.medium,
            levmed=self.levmed,
            valmed=self.valmed,
            ratmed=self.ratmed,
            measure=self.measure,
            S=self.S,
            Pin=self.Pin,
            Pout=self.Pout,
            V2M=self.V2M,
            M2V=self.M2V,
            X=self.X,
            Y=self.Y,
            S_int=self.S_int,
            S_ext=self.S_ext,
            Q=self.Q,
            P=self.P,
            b_int=self.b_int,
            b_ext=self.b_ext,
            Sb=self.Sb,
            c=self.c,
        )

    def load(self, filename):
        # load parameters (npz format)
        if not os.path.isfile(filename + ".npz"):
            print(filename + ".npz")
            sys.exit("file not found")
        loaded = np.load(filename + ".npz")
        self.cobraname = str(loaded["cobraname"])
        self.reduce = str(loaded["reduce"])
        self.reduce = True if self.reduce == "True" else False
        self.mediumname = str(loaded["mediumname"])
        self.mediumbound = str(loaded["mediumbound"])
        self.objective = loaded["objective"]
        self.method = str(loaded["method"])
        self.size = loaded["size"]
        self.medium = loaded["medium"]
        self.levmed = loaded["levmed"]
        self.valmed = loaded["valmed"]
        self.ratmed = loaded["ratmed"]
        self.measure = loaded["measure"]
        self.S = loaded["S"]
        self.Pin = loaded["Pin"]
        self.Pout = loaded["Pout"]
        self.V2M = loaded["V2M"]
        self.M2V = loaded["M2V"]
        self.X = loaded["X"]
        self.Y = loaded["Y"]
        self.S_int = loaded["S_int"]
        self.S_ext = loaded["S_ext"]
        self.Q = loaded["Q"]
        self.P = loaded["P"]
        self.b_int = loaded["b_int"]
        self.b_ext = loaded["b_ext"]
        self.Sb = loaded["Sb"]
        self.c = loaded["c"]
        self.allmatrices = True
        self.model = cobra.io.read_sbml_model(self.cobraname + ".xml")

    def printout(self, filename=""):
        if filename != "":
            sys.stdout = open(filename, "wb")
        print("model file name:", self.cobraname)
        print("reduced model:", self.reduce)
        print("medium file name:", self.mediumname)
        print("medium bound:", self.mediumbound)
        print("list of reactions in objective:", self.objective)
        print("method:", self.method)
        print("trainingsize:", self.size)
        print("list of medium reactions:", len(self.medium))
        print("list of medium levels:", len(self.levmed))
        print("list of medium values:", len(self.valmed))
        print("ratio of variable medium turned on:", self.ratmed)
        print("list of measured reactions:", len(self.measure))
        print("Stoichiometric matrix", self.S.shape)
        print("Boundary matrix from reactions to medium:", self.Pin.shape)
        print("Measurement matrix from reaction to measures:", self.Pout.shape)
        print("Reaction to metabolite matrix:", self.V2M.shape)
        print("Metabolite to reaction matrix:", self.M2V.shape)
        print("Training set X:", self.X.shape)
        print("Training set Y:", self.Y.shape)
        if self.allmatrices:
            print("S_int matrix", self.S_int.shape)
            print("S_ext matrix", self.S_ext.shape)
            print("Q matrix", self.Q.shape)
            print("P matrix", self.P.shape)
            print("b_int vector", self.b_int.shape)
            print("b_ext vector", self.b_ext.shape)
            print("Sb matrix", self.Sb.shape)
            print("c vector", self.c.shape)
        if filename != "":
            sys.stdout.close()

    def get(self, sample_size=100, varmed=[], reduce=False, verbose=False):
        # Generate a training set for AMN
        # Input: sample size
        # objective_value and variable medium
        # (optional when experimental datafile)
        # Output: X,Y (medium and reaction flux values)

        X, Y, inf = {}, {}, {}
        for i in range(sample_size):
            if verbose:
                print("sample:", i)

            # Cobra is run on reduce model where X is already know
            if reduce:
                inf = {r.id: 0 for r in self.model.reactions}
                for j in range(len(self.medium)):
                    inf[self.medium[j]] = self.X[i, j]

            X[i], Y[i] = get_io_cobra(
                self.model,
                self.objective,
                self.medium,
                self.mediumbound,
                varmed,
                self.levmed,
                self.valmed,
                self.ratmed,
                self.Pout,
                inf=inf,
                method=self.method,
                verbose=verbose,
            )
        X = np.asarray(list(X.values()))
        Y = np.asarray(list(Y.values()))

        # In case mediumbound is 'EB' replace X[i] by Y[i] for i in medium
        if self.mediumbound == "EB":
            i = 0
            for rid in self.medium:
                j = get_index_from_id(rid, self.model.reactions)
                X[:, i] = Y[:, j]
                i += 1

        # In case 'get' is called several times
        if self.X.shape[0] > 0 and reduce is False:
            self.X = np.concatenate((self.X, X), axis=0)
            self.Y = np.concatenate((self.Y, Y), axis=0)
        else:
            self.X, self.Y = X, Y
        self.size = self.X.shape[0]

    def filter_measure(self, measure=[], verbose=False):
        # Keep only reaction fluxes in measure
        # Input:
        # - measure: a list of measured reaction fluxes
        # - reduce: when True the matrices are reduced considering
        #   the training set, all reactions not in the medium and
        #   having zero flux for all instances in the trainig set
        #   are removed
        # Output:
        # - updated self.Y (reduced to reaction fluxes in measure)
        # - self.Yall all reactions

        self.measure = measure if len(self.measure) > 0 else self.measure
        _, _, self.Pout, _, _ = get_matrices(
            self.model, self.medium, self.measure, self.model.reactions
        )
        self.Yall = self.Y.copy()
        if self.measure != []:
            # Y = only the reaction fluxes that are in Vout
            Y = np.matmul(self.Y, np.transpose(self.Pout)) if ("EXP") not in self.method else self.Y
            self.Y = Y
        if verbose:
            print("number of reactions: ", self.S.shape[1], self.Yall.shape[1])
            print("number of metabolites: ", self.S.shape[0])
            print("filtered measurements size: ", self.Y.shape[1])


###############################################################################
# This library provide utilities for buiding, training, evaluating, saving
# and loading models. The actual model is passed through the parameter
# 'model_type'. The library makes use of Keras, tensorfow and sklearn
# The provided models are:
# - ANN_dense: a simple Dense neural network
# - AMN_QP: a trainable QP solver using Gradient Descent
# - AMN_LP: a trainable LP solver of primal and dual LP from Y. Yang et al.
#           Mathematics and Computers in Simulation, 101 (2014) 103–112
# - AMN_Wt: a trainable RNN cell where V is updated with a weight matrix
# - MM_QP and MM_LP: non-trainable mechanistic model based on linear program
#   and gradient descent to compute all fluxes V when target objectives
#   are provided
# - RC: make use of trained AMNs (cf. previous module)
#   in reseroir computing (RC). The reservoir (non-trainable AMN)
#   is squized between two standard ANNs. The purpose of the prior ANN is to
#   transform problem features into nutrients added to media.
#   The post-ANN reads reservoir output (user predefined specific
#   reaction rates) and produce a readout to best match training set values.
#   Note that the two ANNs are trained but not the reservoir (AMN).
# Authors: Jean-loup Faulon, jfaulon@gmail.com and Bastien Mollet (LP model)
###############################################################################



def sharp_sigmoid(x):
    # Custom activation function
    return K.sigmoid(10000.0 * x)


get_custom_objects().update({"sharp_sigmoid": Activation(sharp_sigmoid)})


def my_mse(y_true, y_pred):
    # Custom loss function
    end = y_true.shape[1]
    return keras.losses.mean_squared_error(y_true[:, :end], y_pred[:, :end])


def my_mae(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    return keras.losses.mean_squared_error(y_true[:, :end], y_pred[:, :end])


def my_binary_crossentropy(y_true, y_pred):
    # Custom loss function
    end = y_true.shape[1]
    return keras.losses.binary_crossentropy(y_true[:, :end], y_pred[:, :end])


def my_acc(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    return keras.metrics.binary_accuracy(y_true[:, :end], y_pred[:, :end])


def my_r2(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    yt, yp = y_true[:, :end], y_pred[:, :end]
    SS = K.sum(K.square(yt - yp))
    ST = K.sum(K.square(yt - K.mean(yt)))
    return 1 - SS / (ST + K.epsilon())


def CROP(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call x = crop(2,5,10)(x) to slice the second dimension
    def func(x):
        if dimension == 0:
            return x[start:end]
        if dimension == 1:
            return x[:, start:end]
        if dimension == 2:
            return x[:, :, start:end]
        if dimension == 3:
            return x[:, :, :, start:end]
        if dimension == 4:
            return x[:, :, :, :, start:end]

    return Lambda(func)


###############################################################################
# Custom Loss functions to evaluate models and compute gradients
# Inputs:
# - V: the (predicted) flux vector
# - Pout: the matrix selecting in V measured outgoing fluxes
# - Vout: the measured outgoing fluxes
# - Pin: the matrix selecting in V measured incoming fluxes
# - Vin: the measured incoming fluxes
# - S: the stoichiometric matrix
# Outputs:
# - Loss and gradient
###############################################################################

NBR_CONSTRAINT = 3  # The number of contraints of the mechanistic models


def Loss_Vout(V, Pout, Vout, gradient=False):
    # Loss for the objective (match Vout)
    # Loss = ||Pout.V-Vout||
    # When Vout is empty just compute Pout.V
    # dLoss = ∂([Pout.V-Vout]^2)/∂V = Pout^T (Pout.V - Vout)
    Pout = tf.convert_to_tensor(np.float32(Pout))
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True) / Pout.shape[0]
    if gradient:
        dLoss = tf.linalg.matmul(Loss, Pout, b_is_sparse=True)  # derivate
        dLoss = dLoss / (Pout.shape[0] * Pout.shape[0])  # rescaling
        # dLoss = 2 * dLoss
    else:
        dLoss = 0 * V
    return Loss_norm, dLoss


def Loss_SV(V, S, gradient=False):
    # Gradient for SV constraint
    # Loss = ||SV||
    # dLoss =  ∂([SV]^2)/∂V = S^T SV
    S = tf.convert_to_tensor(np.float32(S))
    Loss = tf.linalg.matmul(V, tf.transpose(S), b_is_sparse=True)
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True) / S.shape[0]  # rescaled
    if gradient:
        dLoss = tf.linalg.matmul(Loss, S, b_is_sparse=True)  # derivate
        dLoss = dLoss / (S.shape[0] * S.shape[0])  # rescaling
        dLoss = dLoss / 2
    else:
        dLoss = 0 * V
    return Loss_norm, dLoss


def Loss_Vin(V, Pin, Vin, bound, gradient=False):
    # Gradient for input boundary constraint
    # Loss = ReLU(Pin . V - Vin)
    # dLoss = ∂(ReLU(Pin . V - Vin)^2/∂V
    # Input: Cf. Gradient_Descent
    Pin = tf.convert_to_tensor(np.float32(Pin))
    Loss = tf.linalg.matmul(V, tf.transpose(Pin), b_is_sparse=True) - Vin
    Loss = tf.keras.activations.relu(Loss) if bound == "UB" else Loss
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True) / Pin.shape[0]  # rescaled
    if gradient:
        dLoss = tf.math.divide_no_nan(Loss, Loss)  # derivate: Hadamard div.
        dLoss = tf.math.multiply(Loss, dLoss)  # !!!
        dLoss = tf.linalg.matmul(dLoss, Pin, b_is_sparse=True)  # resizing
        dLoss = dLoss / (Pin.shape[0] * Pin.shape[0])  # rescaling
    else:
        dLoss = 0 * V
    return Loss_norm, dLoss


def Loss_Vpos(V, parameter, gradient=False):
    # Gradient for V ≥ 0 constraint
    # Loss = ReLU(-V)
    # dLoss = ∂(ReLU(-V)^2/∂V
    Loss = tf.keras.activations.relu(-V)
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True) / V.shape[1]  # rescaled
    if gradient:
        dLoss = -tf.math.divide_no_nan(Loss, Loss)  # derivate: Hadamard div.
        dLoss = tf.math.multiply(Loss, dLoss)  # !!!
        dLoss = dLoss / (V.shape[1] * V.shape[1])  # rescaling
    else:
        dLoss = 0 * V
    return Loss_norm, dLoss


def Loss_constraint(V, Vin, parameter, gradient=False):
    # mean squared sum L2+L3+L4
    L2, dL2 = Loss_SV(V, parameter.S, gradient=gradient)
    L3, dL3 = Loss_Vin(V, parameter.Pin, Vin, parameter.mediumbound, gradient=gradient)
    L4, dL4 = Loss_Vpos(V, parameter, gradient=gradient)
    # square sum of L2, L3, L4
    L2 = tf.math.square(L2)
    L3 = tf.math.square(L3)
    L4 = tf.math.square(L4)
    L = tf.math.reduce_sum(tf.concat([L2, L3, L4], axis=1), axis=1)
    # divide by 3
    L = tf.math.divide_no_nan(L, tf.constant(3.0, dtype=tf.float32))
    return L, dL2 + dL3 + dL4


def Loss_all(V, Vin, Vout, parameter, gradient=False):

    # mean square sum of L1, L2, L3, L4
    if Vout.shape[0] < 1:  # No target provided = no Loss_Vout
        L, dL = Loss_constraint(V, Vin, parameter, gradient=gradient)
        return L, dL
    L1, dL1 = Loss_Vout(V, parameter.Pout, Vout, gradient=gradient)
    L2, dL2 = Loss_SV(V, parameter.S, gradient=gradient)
    L3, dL3 = Loss_Vin(V, parameter.Pin, Vin, parameter.mediumbound, gradient=gradient)
    L4, dL4 = Loss_Vpos(V, parameter, gradient=gradient)
    # square sum of L1, L2, L3, L4
    L1 = tf.math.square(L1)
    L2 = tf.math.square(L2)
    L3 = tf.math.square(L3)
    L4 = tf.math.square(L4)
    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
    # divide by 4
    L = tf.math.divide_no_nan(L, tf.constant(4.0, dtype=tf.float32))
    return L, dL1 + dL2 + dL3 + dL4


###############################################################################
# Dense model
###############################################################################


def input_ANN_Dense(parameter, verbose=False):
    # Shape X and Y depending on the model used
    if parameter.scaler != 0:  # Normalize X
        parameter.X, parameter.scaler = MaxScaler(parameter.X)
    if verbose:
        print("ANN Dense scaler", parameter.scaler)
    return parameter.X, parameter.Y


def Dense_layers(inputs, parameter, trainable=True, verbose=False):
    # Build a dense architecture with some hidden layers

    activation = parameter.activation
    n_hidden = parameter.n_hidden
    dropout = parameter.dropout
    hidden_dim = parameter.hidden_dim
    output_dim = parameter.output_dim
    hidden = inputs
    n_hidden = 0 if hidden_dim == 0 else n_hidden
    INITIALIZER = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    for i in range(n_hidden):
        hidden = Dense(
            hidden_dim,
            kernel_initializer=INITIALIZER,
            bias_initializer="zeros",
            activation="relu",
            trainable=trainable,
        )(hidden)
        hidden = Dropout(dropout)(hidden)
    if verbose:
        print(
            "Dense layer n_hidden, hidden_dim, output_dim, activation, trainable:",
            n_hidden,
            hidden_dim,
            output_dim,
            activation,
            trainable,
        )
    outputs = Dense(
        output_dim,
        kernel_initializer="random_normal",
        bias_initializer="zeros",
        activation=activation,
        trainable=trainable,
    )(hidden)
    return outputs


def ANN_Dense(parameter, trainable=True, verbose=False):
    # A standard Dense model with several layers

    input_dim, output_dim = parameter.input_dim, parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs = Dense_layers(inputs, parameter, trainable=trainable, verbose=verbose)
    model = keras.models.Model(inputs=[inputs], outputs=[outputs])
    loss = "mse" if parameter.regression else "binary_crossentropy"
    metrics = ["mae"] if parameter.regression else ["acc"]
    print("Parameter regression =", parameter.regression)
    model.compile(loss=loss, optimizer="adam", metrics=metrics)
    if verbose == 2:
        print(model.summary())
    print("nbr parameters:", model.count_params())
    parameter.model = model

    return parameter


###############################################################################
# AMN models (1)
# AMN_QP: a ANN_Dense trainable prior layer and a mechanistic layer
# making use of gradient descent
###############################################################################


def input_AMN(parameter, verbose=False):
    # Shape the IOs
    # IO: X and Y
    # For all
    # - add additional zero columns to Y
    #   the columns are used to minimize SV, Pin V ≤ Vin, V ≥ 0
    # For AMN_LP: add b_int or b_ext
    # For AMN_Wt repeat X timestep times

    X, Y = parameter.X, parameter.Y
    if parameter.scaler != 0:  # Normalize X
        X, parameter.scaler = MaxScaler(X)
    if verbose:
        print("AMN scaler", parameter.scaler)
    y = np.zeros(Y.shape[0]).reshape(Y.shape[0], 1)
    Y = np.concatenate((Y, y), axis=1)  # SV constraint
    Y = np.concatenate((Y, y), axis=1)  # Pin constraint
    Y = np.concatenate((Y, y), axis=1)  # V ≥ 0 constraint
    if "QP" in parameter.model_type:
        if verbose:
            print("QP input shape", X.shape, Y.shape)
    elif "RC" in parameter.model_type:
        if verbose:
            print("RC input shape", X.shape, Y.shape)
    elif "LP" in parameter.model_type:
        # we add b_int and b_ext
        x = parameter.b_int
        x = np.vstack([x] * X.shape[0]) if len(x.shape) == 1 else x
        b_int = np.copy(x)
        X = np.concatenate((X, x), axis=1)
        x = parameter.b_ext
        x = np.vstack([x] * X.shape[0]) if len(x.shape) == 1 else x
        b_ext = np.copy(x)
        X = np.concatenate((X, x), axis=1)
        if parameter.mediumbound == "UB":
            parameter.b_int, parameter.b_ext = b_int, b_ext
        else:  # EB
            parameter.b_int, parameter.b_ext = b_ext, b_int
        if verbose:
            print("LP input shape", X.shape, Y.shape)
    elif "Wt" in parameter.model_type:
        x = np.copy(X)
        num_batches = int(x.shape[0] / parameter.batch_size)
        X = np.zeros(
            (parameter.batch_size * num_batches, parameter.timestep, x.shape[1])
        )
        for i in range(x.shape[0]):
            for j in range(parameter.timestep):
                X[i][j] = x[i]
        if verbose:
            print("Wt input shape", X.shape, Y.shape)
    else:
        print(parameter.model_type)
        sys.exit("This AMN type does not have input")
    parameter.input_dim = parameter.X.shape[1]

    return X, Y


def output_AMN(V, Vin, V0, parameter, verbose=False):
    # Get output for all AMN models
    # output = PoutV + constaints = [SV + PinV + Relu(_V)] + V
    # where S and Pout are the stoichiometric and measurement matrix

    Pout = tf.convert_to_tensor(np.float32(parameter.Pout))
    PoutV = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True)
    SV, _ = Loss_SV(V, parameter.S)  # SV const
    PinV, _ = Loss_Vin(V, parameter.Pin, Vin, parameter.mediumbound)  # Pin
    Vpos, _ = Loss_Vpos(V, parameter)  # V ≥ 0 const

    # Return outputs = PoutV + SV + PinV + Vpos + V
    if V0 is None:
        outputs = concatenate([PoutV, SV, PinV, Vpos, V], axis=1)
    else:
        outputs = concatenate([PoutV, SV, PinV, Vpos, V, V0], axis=1)
    parameter.output_dim = outputs.shape[1]
    if verbose:
        print(
            "AMN output shapes for PoutV, SV, PinV, Vpos, V, outputs",
            PoutV.shape,
            SV.shape,
            PinV.shape,
            Vpos.shape,
            V.shape,
            outputs.shape,
        )

    return outputs


def Gradient_Descent(
    V, Vin, Vout, parameter, mask, trainable=True, history=False, verbose=False
):
    # Input:
    # S [m x n]: stoichiometric matrix
    # V [n]: the reaction flux vector
    # Pin [n_in x n]: the flux to medium projection matrix
    # Vin [p]: the medium intake flux vector
    # V_out [n_out]: the measured fluxes (can be empty)
    # mask [n]: used to uddate dL
    # history: to specify if loss is computed and recorded
    # Output: Loss and updated V

    # Not history here if trainable
    history = False if trainable else history

    # GD loop
    Loss_mean_history, Loss_std_history, diff = [], [], 0 * V
    for t in range(1, parameter.timestep + 1):  # Update V with GD
        # Get Loss and gradient
        L, dL = Loss_all(V, Vin, Vout, parameter, gradient=True)
        dL = tf.math.multiply(dL, mask)  # Apply mask on dL
        # Update V with learn and decay rates
        diff = parameter.decay_rate * diff - parameter.learn_rate * dL
        V = V + diff
        # Compile Loss history
        if history:
            Loss_mean, Loss_std = np.mean(L), np.std(L)
            Loss_mean_history.append(Loss_mean)
            Loss_std_history.append(Loss_std)
            if verbose and (
                np.log10(t) == int(np.log10(t)) or t / 1.0e3 == int(t / 1.0e3)
            ):
                print("QP-Loss", t, Loss_mean, Loss_std)

    return V, Loss_mean_history, Loss_std_history


def get_V0(inputs, parameter, targets, trainable, verbose=False):
    # Get initial vector V0 from input and target
    # Return V0, Vin, Vout, mask
    # When target is not provided this function compute
    # the initial vector V0 using Dense_Layers

    Pin = tf.convert_to_tensor(np.float32(parameter.Pin))
    if targets.shape[0] > 0:  # Initialize AMN when targets provided
        # Vin = inputs, V0 = (Pin)^T Vin
        Vin = inputs
        V0 = tf.linalg.matmul(inputs, Pin, b_is_sparse=True)
    else:  # Initialize AMN when targets not provided
        # Vin = inputs, V0 = Dense_layers(inputs)
        param = copy.copy(parameter)
        param.output_dim = parameter.S.shape[1]
        param.activation = "relu"
        Vin = inputs
        V0 = Dense_layers(inputs, param, trainable=trainable, verbose=verbose)

    # Get a mask for EB and UB where elements in Vin are not updated in V
    ones = np.ones(parameter.S.shape[1])
    ones = tf.convert_to_tensor(np.float32(ones))
    # mask = np.matmul(np.ones(Vin.shape[1]), Pin)
    mask = tf.linalg.matvec(Pin, tf.ones([Vin.shape[1]]), transpose_a=True)
    # element in Vin are at 0 in mask others are at 1
    mask = ones - mask

    # Vin projection in V: elements not in Vin are at 0
    VinV = tf.linalg.matmul(Vin, Pin, b_is_sparse=True)
    if parameter.mediumbound == "UB":  # we must have V ≤ Vin
        # relu = 1 when VinV > V, 0 othervise
        relu = tf.keras.activations.relu(VinV - V0)
        relu = tf.math.divide_no_nan(relu, relu)  # 0/1 tensor
        # VinV = V when V < Vin, VinV = Vin when V > Vin
        VinV = relu * V0 + (ones - relu) * VinV
    V0 = tf.math.multiply(V0, mask) + VinV
    Vout = tf.convert_to_tensor(np.float32(targets))
    mask = ones if parameter.mediumbound == "UB" else mask

    return V0, Vin, Vout, mask


def QP_layers(
    inputs,
    parameter,
    targets=np.asarray([]).reshape(0, 0),
    trainable=True,
    history=False,
    verbose=False,
):
    # Build and return an architecture using GD
    # The function is used with and without targets
    # - With targets there is no training set and GD is run
    #   to optimize both the objective min([PV-Target]^2))
    #   and the constraints.
    # - Without target an initial vector V is calculated via training
    #   through a Dense layer, GD is only used
    #   to minimize the constrains
    # Inputs:
    # - input flux vector, targets (can be empty)
    # - flags to train, record Loss history
    # Outputs:
    # - ouput_AMN (see function, and Loss (mean and std)

    V0, Vin, Vout, mask = get_V0(inputs, parameter, targets, trainable, verbose=verbose)
    V, Loss_mean, Loss_std = Gradient_Descent(
        V0,
        Vin,
        Vout,
        parameter,
        mask,
        trainable=trainable,
        history=history,
        verbose=verbose,
    )
    outputs = output_AMN(V, Vin, V0, parameter, verbose=verbose)

    return outputs, Loss_mean, Loss_std


def AMN_QP(parameter, trainable=True, verbose=False):
    # Build and return an AMN with training
    # input : problem parameter
    # output: Trainable model
    # Loss history is not recorded (already done thru tf training)

    # Get dimensions and build model
    input_dim, output_dim = parameter.X.shape[1], parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs, loss_h, loss_std_h = QP_layers(
        inputs, parameter, trainable=trainable, history=False, verbose=verbose
    )
    # Compile
    model = keras.models.Model(inputs=[inputs], outputs=outputs)
    (loss, metrics) = (my_mse, [my_r2])
    model.compile(loss=loss, optimizer="adam", metrics=metrics)
    if verbose == 2:
        print(model.summary())
    print("nbr parameters:", model.count_params())
    parameter.model = model

    return parameter


###############################################################################
# AMN models (2)
# AMN_LP: a ANN_Dense trainable prior layer and a mechanistic layer
# making use of the "LP" method (Solving a linear program with a
# recurrent neural network)
# Code written by Bastien Mollet
###############################################################################


def LP(V, M, b_int, b_ext, parameter, verbose=False):
    # Inputs:
    # - b_int and b_ext are not the same depending on whether
    #   UB is true or not (more details after).
    # Outputs: dV, dM
    # Recurrent cell from Y. Yang et al.
    # Mathematics and Computers in Simulation, 101 (2014) 103–112

    # Format all matrices and vectors
    OBJECTIVE_SCALER = 100  # scaler for the objective function c
    S_int, S_ext, Q, P, Sb, c = (
        tf.convert_to_tensor(parameter.S_int),
        tf.convert_to_tensor(parameter.S_ext),
        tf.convert_to_tensor(parameter.Q),
        tf.convert_to_tensor(parameter.P),
        tf.convert_to_tensor(parameter.Sb),
        tf.convert_to_tensor(OBJECTIVE_SCALER * parameter.c),
    )

    # The main difference between EB and UB is on the precomputed matrixes
    # S_int, S_ext, Q, P, Sb and the b_int and b_ext vectors
    SV = tf.linalg.matvec(S_ext, V, transpose_a=True)  # R = (M + S_extV - b_ext)^+
    R = tf.keras.activations.relu(M + SV - b_ext)
    dV1 = tf.linalg.matvec(S_ext, R) + c  # dV1 = P(S_extR + c)
    dV1 = tf.linalg.matvec(P, dV1)
    dV2 = tf.linalg.matvec(S_int, V)  # dV2 = Q(S_intV - b_int)
    dV2 = dV2 - b_int
    dV2 = tf.linalg.matvec(Q, dV2)
    dV = dV1 - dV2  # dV
    dM = 0.5 * (R - M)  # dM = 1/2 (R - M)
    return dV, dM


def get_M0(inputs, parameter, targets, trainable, verbose=False):
    # Get initial vectors M0 from inputs and target
    # M0 is the initial value of the dual variable of LP
    # Return M0
    # When targets is provided M0 = 0 vector
    # When target is not provided M0 is computed via training
    # of Dense_Layers

    if parameter.mediumbound == "UB":
        M0_size = 2 * parameter.S.shape[0] + parameter.S.shape[1]
    else:
        M0_size = parameter.S.shape[1]
    if targets.shape[0] > 0:  # for MM models
        # Initialize M0 = 0 when targets provided, for MM (solving)
        M0 = tf.zeros((targets.shape[0], M0_size), dtype=tf.float32)
    else:  # AMN models
        # M0 = Dense_layers(inputs)
        param = copy.copy(parameter)
        param.output_dim = M0_size
        param.activation = "linear"  # M0 can be negative
        M0 = Dense_layers(inputs, param, trainable=trainable, verbose=verbose)
    return M0


def LP_layers(
    inputs_bounds,
    parameter,
    targets=np.asarray([]).reshape(0, 0),
    trainable=True,
    history=False,
    verbose=False,
):
    # Build and return an architecture using LP
    # UB:
    # Here the dimension of M corresponds to the 3 constraints:
    # 1: S_ext*V < b_ext [meta_dim] Upper bounds
    #    that are set from the medium compounds.
    # 2: V > 0 [flux_dim] Fluxs are supposed positive as we had split them.
    # 3: S_ext*V > 0 [meta_dim] ensure
    #    that there is no metabolite leaking from the cell.
    #    Thus the dimension of m is [flux_dim + 2*metadim]
    # EB :
    # the only inequality is V > 0 (that's why S_ext = I [n.n])
    # Inputs:
    # - input flux vector + boudary fluxes, targets (can be empty)
    # - flags to train, record Loss history
    # Outputs:
    # - ouput_AMN (see function, and Loss (mean and std)

    # Not history here if trainable
    history = False if trainable else history

    # Initialize AMN with VO, M0, b_int, b_ext
    inputs = CROP(1, 0, parameter.Pin.shape[0])(inputs_bounds)
    b_int = CROP(
        1, parameter.Pin.shape[0], parameter.Pin.shape[0] + parameter.b_int.shape[1]
    )(inputs_bounds)
    b_ext = CROP(
        1,
        parameter.Pin.shape[0] + parameter.b_int.shape[1],
        parameter.Pin.shape[0] + parameter.b_int.shape[1] + parameter.b_ext.shape[1],
    )(inputs_bounds)
    print(inputs, b_int, b_ext)
    V0, Vin, Vout, mask = get_V0(inputs, parameter, targets, trainable, verbose=verbose)
    M0 = get_M0(inputs, parameter, targets, trainable, verbose=verbose)

    # LP loop
    V, M = V0, M0
    Loss_mean_history, Loss_std_history = [], []
    for t in range(1, parameter.timestep + 1):
        # Get Loss and gradients
        L, _ = Loss_all(V, Vin, Vout, parameter)
        dV, dM = LP(V, M, b_int, b_ext, parameter, verbose=verbose)
        dV = tf.math.multiply(dV, mask)  # Apply mask on dV
        V = V + parameter.learn_rate * dV
        M = M + parameter.learn_rate * dM
        # Compile Loss history
        if history:
            Loss_mean, Loss_std = np.mean(L), np.std(L)
            Loss_mean_history.append(Loss_mean)
            Loss_std_history.append(Loss_std)
            if verbose and (
                np.log10(t) == int(np.log10(t)) or t / 1.0e3 == int(t / 1.0e3)
            ):
                print("LP-Loss", t, Loss_mean, Loss_std)
    outputs = output_AMN(V, Vin, V0, parameter, verbose=verbose)

    return outputs, Loss_mean_history, Loss_std_history


def AMN_LP(parameter, trainable=True, verbose=False):
    # Build and return an AMN with training
    # input : problem parameter
    # output: Trainable model
    # Loss history is not recorded (already done thru tf training)

    # Get dimensions and build model
    input_dim, output_dim = parameter.X.shape[1], parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs, loss_h, loss_std_h = LP_layers(
        inputs, parameter, trainable=trainable, history=False, verbose=verbose
    )
    # Compile
    model = keras.models.Model(inputs=[inputs], outputs=outputs)
    (loss, metrics) = (my_mse, [my_r2])
    model.compile(loss=loss, optimizer="adam", metrics=metrics)
    if verbose == 2:
        print(model.summary())
    print("nbr parameters:", model.count_params())
    parameter.model = model
    return parameter


###############################################################################
# AMN models (3)
# AMN_Wt: An RNN where input (the medium) and flux vector V are passed
# to the recurrent cell
# M = V2M . V
# V = Win x Vin + Wrec x M2V . M
# Win and Wrec are weight matrices learned during training
# A hidden layer can be added to Win (not Wrec)
# Warning: The model AMN_Wt works only with UB training sets
###############################################################################


class RNNCell(keras.layers.Layer):  # RNN Cell, as a layer subclass.
    def __init__(self, parameter):
        meta_dim = parameter.S.shape[0]
        flux_dim = parameter.S.shape[1]
        medm_dim = parameter.Pin.shape[0]
        self.input_size = medm_dim
        self.state_size = flux_dim
        self.mediumbound = parameter.mediumbound
        self.hidden_dim = parameter.hidden_dim
        self.S = tf.convert_to_tensor(np.float32(parameter.S))
        self.V2M = tf.convert_to_tensor(np.float32(parameter.V2M))
        self.Pin = tf.convert_to_tensor(np.float32(parameter.Pin))
        # Normalize M2V
        M2V = parameter.M2V
        for i in range(flux_dim):
            if np.count_nonzero(M2V[i]) > 0:
                M2V[i] = M2V[i] / np.count_nonzero(M2V[i])
        self.M2V = tf.convert_to_tensor(np.float32(M2V))
        self.dropout = parameter.dropout
        super(RNNCell, self).__init__(True)

    def build(self, input_shape):
        meta_dim = self.S.shape[0]
        flux_dim = self.S.shape[1]
        medm_dim = self.input_size
        hidden_dim = self.hidden_dim
        # weigths to compute V for both input (i) and recurrent cell (r)
        if self.mediumbound == "UB":  # no kernel_Vh and kernel_Vi for EB
            if hidden_dim > 0:  # plug an hidden layer upstream of Winput
                self.wh_V = self.add_weight(
                    shape=(medm_dim, hidden_dim), name="kernel_Vh"
                )
                self.wi_V = self.add_weight(
                    shape=(hidden_dim, medm_dim), name="kernel_Vi"
                )
            else:
                self.wi_V = self.add_weight(
                    shape=(medm_dim, medm_dim), name="kernel_Vi"
                )
        self.wr_V = self.add_weight(shape=(flux_dim, meta_dim), name="kernel_Vr")
        self.bi_V = self.add_weight(
            shape=(medm_dim,),
            initializer="random_normal",
            name="bias_Vi",
            trainable=True,
        )
        self.br_V = self.add_weight(
            shape=(flux_dim,),
            initializer="random_normal",
            name="bias_Vr",
            trainable=True,
        )
        self.built = True

    def call(self, inputs, states):
        # At steady state we have
        # M = V2M V and V = (M2V x W) M + V0
        V = states[0]
        if self.mediumbound == "UB":
            if self.hidden_dim > 0:
                VH = K.dot(inputs, self.wh_V)
                V0 = K.dot(VH, self.wi_V) + self.bi_V
            else:
                V0 = K.dot(inputs, self.wi_V) + self.bi_V
        else:
            V0 = inputs  # EB case
        V0 = tf.linalg.matmul(V0, self.Pin, b_is_sparse=True)
        M = tf.linalg.matmul(V, tf.transpose(self.V2M), b_is_sparse=True)
        W = tf.math.multiply(self.M2V, self.wr_V)
        V = tf.linalg.matmul(M, tf.transpose(W), b_is_sparse=True)
        V = V + V0 + self.br_V
        return V, [V]

    def get_config(self):  # override tf.get_config to save RNN model
        # The code below does not work !! anyone to debug?
        config = super().get_config().copy()
        # config.update({'parameter': self.parameter.__dict__})
        return config


def Wt_layers(inputs, parameter, trainable=True, verbose=False):
    # Build and return AMN layers using an RNN cell
    with CustomObjectScope({"RNNCell": RNNCell}):
        rnn = keras.layers.RNN(RNNCell(parameter))
    V = rnn(inputs)
    Vin = inputs[:, 0, :]
    return output_AMN(V, Vin, None, parameter, verbose=verbose)


def AMN_Wt(parameter, verbose=False):
    # Build and return an AMN using an RNN cell
    # input : medium vector in parameter
    # output: experimental steaty state fluxes

    # Get dimensions and build model
    input_dim, output_dim = parameter.X.shape[2], parameter.Y.shape[1]
    inputs = keras.Input((None, input_dim))
    outputs = Wt_layers(inputs, parameter)

    # Compile
    model = keras.models.Model(inputs, outputs)
    (loss, metrics) = (my_mse, [my_r2])
    model.compile(loss=loss, optimizer="adam", metrics=metrics)
    if verbose == 2:
        print(model.summary())
    print("nbr parameters:", model.count_params())
    parameter.model = model

    return parameter


###############################################################################
# Non-trainable Mechanistic Model (MM)
# using QP or LP
###############################################################################


def write_loss(f_name, param, mean_history, std_history):
    if f_name is None:
        return 0
    timesteps = np.array(range(1, param.timestep + 1))
    losses = np.array(mean_history)
    stdevs = np.array(std_history)
    to_write = np.concatenate(
        [
            timesteps.reshape((len(timesteps), 1)),
            losses.reshape((len(losses), 1)),
            stdevs.reshape((len(stdevs), 1)),
        ],
        axis=1,
    )
    np.savetxt(f_name, to_write, delimiter=",")
    return 0


def write_targets(f_name, param, Ypred):
    if f_name is None:
        return 0
    true = np.array(param.Y)
    pred = np.array(Ypred)
    to_write = np.concatenate(
        [true.reshape((len(true), 1)), pred.reshape((len(pred), 1))], axis=1
    )
    np.savetxt(f_name, to_write, delimiter=",")
    return 0


def get_flux_output(param, output):
    # Just getting vector V from output
    # output : PoutV (=Ypred) + SV + PinV + Vpos + V + V0
    len_fluxes = param.S.shape[1]
    if output.shape[1] > (len_fluxes + NBR_CONSTRAINT + 1):  # case where we get V0
        V0 = CROP(
            1,
            param.Y.shape[1] + NBR_CONSTRAINT + len_fluxes,
            param.Y.shape[1] + NBR_CONSTRAINT + len_fluxes * 2,
        )(output)
        Vf = CROP(
            1,
            param.Y.shape[1] + NBR_CONSTRAINT,
            param.Y.shape[1] + NBR_CONSTRAINT + len_fluxes,
        )(output)
    else:  # case where we don't have V0 at the end of the output
        Vf = CROP(
            1,
            param.Y.shape[1] + NBR_CONSTRAINT,
            param.Y.shape[1] + NBR_CONSTRAINT + len_fluxes,
        )(output)
    return Vf


def MM_LP_QP(
    parameter,
    LP=True,
    loss_outfile=None,
    targets_outfile=None,
    history=True,
    verbose=False,
):
    # Solve LP or QP without training
    # inputs:
    # - problem parameter, history flag
    # output:
    # - Predicted all fluxes and stats = loss history

    # inputs must be in tf format
    param = copy.copy(parameter)
    if param.X.shape[1] < param.S.shape[1]:
        # when all X provided no need to tranform
        param.X, _ = input_AMN(param, verbose=False)
    inputs = tf.convert_to_tensor(np.float32(param.X))
    targets = param.Y

    # run LP or QP
    if LP:
        output, Loss_mean, Loss_std = LP_layers(
            inputs,
            param,
            targets=targets,
            trainable=False,
            history=history,
            verbose=verbose,
        )
    else:
        output, Loss_mean, Loss_std = QP_layers(
            inputs,
            param,
            targets=targets,
            trainable=False,
            history=history,
            verbose=verbose,
        )
    Ypred = CROP(1, 0, param.Y.shape[1])(output)
    Vf = get_flux_output(param, output)
    # compute R2 and write losses and targets
    r2 = r2_score(param.Y, Ypred.numpy(), multioutput="variance_weighted")
    write_loss(loss_outfile, parameter, Loss_mean, Loss_std)
    write_targets(targets_outfile, parameter, Ypred)

    return Vf.numpy(), ReturnStats(r2, 0, Loss_mean[-1], Loss_std[-1], 0, 0, 0, 0)


def MM_LP(
    parameter, loss_outfile=None, targets_outfile=None, history=True, verbose=False
):
    # Solve LP without training
    return MM_LP_QP(
        parameter,
        LP=True,
        loss_outfile=loss_outfile,
        targets_outfile=targets_outfile,
        history=history,
        verbose=verbose,
    )


def MM_QP(
    parameter, loss_outfile=None, targets_outfile=None, history=True, verbose=False
):
    # Solve QP without training
    return MM_LP_QP(
        parameter,
        LP=False,
        loss_outfile=loss_outfile,
        targets_outfile=targets_outfile,
        history=history,
        verbose=verbose,
    )


###############################################################################
# RC models
# This module is making use of trained AMNs (cf. previous module)
# in reseroir computing (RC). The reservoir (non-trainable AMN)
# is squized between two standard ANNs. The purpose of the prior ANN is to
# transform problem features into nutrients added to media.
# The post-ANN reads reservoir output (user predefined specific
# reaction rates) and produce a readout to best match training set values.
# Note that the two ANNs are trained but not the reservoir (AMN).
###############################################################################


def input_RC(parameter, verbose=False):
    # Shape X and Y depending on the model used
    if "AMN" in parameter.model_type:
        return input_AMN(parameter, verbose=verbose)
    return parameter.X, parameter.Y


def RC(parameter, verbose=False):
    # Build and return a Reservoir Computing model
    # The model is composed of
    # - A prior trainable network that generate
    #   an outpout = input of the reservoir
    # - The non-trainable reservoir (must have been created and saved)
    # - A post trainable network that takes as input the reservor output
    #   and produce problem's output
    # - A last layer that concatenate the prior trainable output
    #   and the post trainable output

    # Prior
    # If the mode of the reservoir is UB prior computes only
    # the variable part of the medium
    inputs, L = Input(shape=(parameter.input_dim,)), 0
    if parameter.prior:
        if parameter.res.mediumbound == "UB":
            L = inputs.shape[1] - parameter.prior.input_dim
            Res_inputs = CROP(1, 0, L)(inputs)  # minmed
            Prior_inputs = CROP(1, L, inputs.shape[1])(inputs)  # varmed
        else:  # L=0
            Prior_inputs = inputs
        Prior_outputs = Dense_layers(
            Prior_inputs, parameter.prior, trainable=True, verbose=verbose
        )
        if verbose:
            print("Prior inputs and outputs", Prior_inputs.shape, Prior_outputs.shape)
            print("Res inputs added to Prior_outputs", L)
    else:
        Prior_outputs = inputs

    # Reservoir
    # 1. Add to Res_inputs the fixed part of input
    # 2. Mask in Res_inputs all zero elements in inputs
    # 3. Call (non trainable) reservoir
    # 4. Get for forward passing the loss on constraints
    if L > 0:
        Res_inputs = concatenate([Res_inputs, Prior_outputs], axis=1)
    else:
        Res_inputs = Prior_outputs

    # O/1 mask
    inputs_mask = tf.math.divide_no_nan(inputs, inputs)
    Res_inputs = tf.math.multiply(Res_inputs, inputs_mask)
    # Res inputs is scaled to fit training data
    if parameter.res.scaler != 0:
        Res_inputs = Res_inputs / parameter.res.scaler
    if verbose:
        print("Res inputs (final)", Res_inputs.shape)
    # Run reservoir
    if "AMN" in parameter.model_type:
        Res_layers = QP_layers
    else:
        sys.exit("AMN is the only reservoir type handled with RC")
    Res_outputs, _, _ = Res_layers(
        Res_inputs, parameter.res, trainable=False, verbose=verbose
    )
    if verbose:
        print("Res_outputs--------------------", Res_outputs.shape)
    # Get losses
    L = len(parameter.res.objective)  # Objective length
    Post_inputs = CROP(1, 0, L)(Res_outputs)  # Objective only
    SV = CROP(1, L, L + 1)(Res_outputs)
    PinV = CROP(1, L + 1, L + 2)(Res_outputs)
    Vpos = CROP(1, L + 2, L + 3)(Res_outputs)
    V = CROP(1, L + 3, L + 3 + parameter.res.S.shape[1])(Res_outputs)
    if verbose:
        print(
            "SV, PinV, Vpos, V--------------", SV.shape, PinV.shape, Vpos.shape, V.shape
        )

    # Post
    if parameter.post:
        if verbose:
            print("Post_inputs--------------------", Post_inputs.shape)
        Post_outputs = Dense_layers(Post_inputs, parameter.post, verbose=verbose)
    else:
        Post_outputs = Post_inputs

    # RC output
    outputs = concatenate([Post_outputs, SV, PinV, Vpos, V, Res_inputs], axis=1)

    # Compile optimizer parametized for few data
    model = keras.models.Model(inputs, outputs)
    (loss, metrics) = (
        (my_mse, [my_mae])
        if parameter.regression
        else (my_binary_crossentropy, [my_acc])
    )
    print("Parameter regression =", parameter.regression)
    opt = tf.keras.optimizers.Adam(learning_rate=parameter.train_rate)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    if verbose == 2:
        print(model.summary())
    print("nbr parameters:", model.count_params())
    parameter.model = model

    # Set the weights on the non-trainable part of the network
    weights_model = model.get_weights()
    weights_reservoir = parameter.res.model.get_weights()
    if parameter.post:
        lenpost = 2 + parameter.post.n_hidden * 2  # 2 = weight + biais
    else:
        lenpost = 0
    N = len(weights_model) - len(weights_reservoir) - lenpost
    for i in range(len(weights_reservoir)):
        weights_model[N + i] = weights_reservoir[i]
    model.set_weights(weights_model)
    model.trainable = False

    return parameter


###############################################################################
# Train and Evaluate all models
###############################################################################


class ReturnStats:
    def __init__(self, v1, v2, v3, v4, v5, v6, v7, v8):
        self.train_objective = (v1, v2)
        self.train_loss = (v3, v4)
        self.test_objective = (v5, v6)
        self.test_loss = (v7, v8)


def print_loss_evaluate(y_true, y_pred, Vin, parameter):
    # Print all losses
    loss_out0, loss_outf = -1, -1
    loss_cst0, loss_cstf = -1, -1
    loss_all0, loss_allf = -1, -1
    end = y_true.shape[1] - NBR_CONSTRAINT
    nV = parameter.S.shape[1]
    Vf = y_pred[:, y_true.shape[1] : y_true.shape[1] + nV]
    Vout = y_true[:, :end]
    if y_pred.shape[1] == y_true.shape[1] + nV + nV:
        V0 = y_pred[:, y_true.shape[1] + nV : y_true.shape[1] + nV + nV]
        loss_out0, _ = Loss_Vout(V0, parameter.Pout, Vout)
        loss_out0 = np.mean(loss_out0.numpy())
        loss_cst0, _ = Loss_constraint(V0, Vin, parameter)
        loss_cst0 = np.mean(loss_cst0.numpy())
        loss_all0, _ = Loss_all(V0, Vin, Vout, parameter)
        loss_all0 = np.mean(loss_all0.numpy())
    loss_outf, _ = Loss_Vout(Vf, parameter.Pout, Vout)
    loss_outf = np.mean(loss_outf.numpy())
    loss_cstf, _ = Loss_constraint(Vf, Vin, parameter)
    loss_cstf = np.mean(loss_cstf.numpy())
    loss_allf, _ = Loss_all(Vf, Vin, Vout, parameter)
    loss_allf = np.mean(loss_allf.numpy())
    print("Loss out on V0: ", loss_out0)
    print("Loss constraint on V0: ", loss_cst0)
    print("Loss all on V0: ", loss_all0)
    print("Loss out on Vf: ", loss_outf)
    print("Loss constraint on Vf: ", loss_cstf)
    print("Loss all on Vf: ", loss_allf)
    if y_pred.shape[1] == y_true.shape[1] + nV + nV:
        d = np.linalg.norm(Vf - V0)
        print("Distance V0 to Vf %f: " % (d))
    return


def get_loss_evaluate(x, y_true, y_pred, parameter, verbose=False):
    # Return loss on constraint for y_pred

    if "AMN" in parameter.model_type:
        nV = parameter.S.shape[1]
        Vf = y_pred[:, y_true.shape[1] : y_true.shape[1] + nV]
        if "AMN_LP" in parameter.model_type:
            # x = Vin + bounds is truncated
            Vin = x[:, 0 : parameter.Pin.shape[0]]
        elif "AMN_Wt" in parameter.model_type:
            # The dimension (time) added to x with RNN is removed
            Vin = x[:, 0, :]
        else:
            Vin = x
        if verbose:
            print_loss_evaluate(y_true, y_pred, Vin, parameter)
        loss, _ = Loss_constraint(Vf, Vin, parameter)
        loss = np.mean(loss.numpy())
    else:
        loss = -1

    return loss


def evaluate_model(model, x, y_true, parameter, verbose=False):
    # Return y_pred, stats (R2/Acc) for objective
    # and error on constraints for regression and classification

    y_pred = model.predict(x)  # whole y prediction
    print("########################################################## Y pred ##########################################################", y_pred)
    print("########################################################## len Y pred ##########################################################", len(y_pred))
    print("########################################################## len Y pred[0] ##########################################################", len(y_pred[0]))
    # AMN models have NBR_CONSTRAINT constraints added to y_true
    end = (
        y_true.shape[1] - NBR_CONSTRAINT
        if "AMN" in parameter.model_type
        else y_true.shape[1]
    )
    if parameter.regression:
        yt, yp = y_true[:, :end], y_pred[:, :end]
        if yt.shape[0] == 1:  # LOO case
            rss, tss = (yp - yt) * (yp - yt), yt * yt
            if np.sum(tss) > 0:
                obj = 1 - np.sum(rss) / np.sum(tss)
            else:
                obj = 1 - np.sum(rss)
            print("LOO True, Pred, Q2 =", yt, yp, obj)
        else:
            try:
                obj = r2_score(yt, yp, multioutput="variance_weighted")
            except ValueError:
                obj = -1
    else:
        print("Accuracy")
        obj = keras.metrics.binary_accuracy(y_true[:, :end], y_pred[:, :end]).numpy()
        obj = np.count_nonzero(obj) / obj.shape[0]

    # compute stats on constraints
    loss = get_loss_evaluate(x, y_true, y_pred, parameter, verbose=verbose)
    stats = ReturnStats(obj, 0, loss, 0, obj, 0, loss, 0)

    return y_pred, stats


def model_input(parameter, trainable=True, verbose=False):
    # return input for the appropriate model_type
    if "ANN" in parameter.model_type:
        return input_ANN_Dense(parameter, verbose=verbose)
    elif "AMN" in parameter.model_type:
        return input_AMN(parameter, verbose=verbose)
    elif "RC" in parameter.model_type:
        return input_RC(parameter, verbose=verbose)
    elif "MM" in parameter.model_type:
        return input_AMN(parameter, verbose=verbose)
    else:
        print(parameter.model_type)
        sys.exit("no input available")


def model_type(parameter, verbose=False):
    # create the appropriate model_type
    if "ANN_Dense" in parameter.model_type:
        return ANN_Dense(parameter, verbose=verbose)
    elif "AMN_LP" in parameter.model_type:
        return AMN_LP(parameter, verbose=verbose)
    elif "AMN_QP" in parameter.model_type:
        return AMN_QP(parameter, verbose=verbose)
    elif "AMN_Wt" in parameter.model_type:
        return AMN_Wt(parameter, verbose=verbose)
    elif "RC" in parameter.model_type:
        return RC(parameter, verbose=verbose)
    else:
        print(parameter.model_type)
        sys.exit("not a trainable model")


def train_model(parameter, Xtrain, Ytrain, Xtest, Ytest, verbose=False):
    # A standard function to create a model, fit, and test
    # with early stopping
    # Inptuts:
    # - all necessary parameter including
    #   parameter.model, the function used to create the model
    #   parameter.input_model, the function used to shape the model inputs
    #   parameter.X and parameter.Y, the dataset
    #   parameter.regression (boolean) if false classification
    # Outputs:
    # - Net: the trained network
    # - ytrain, ytest: y values for training and tets sets
    # - otrain, ltrain: objective and loss for trainig set
    # - otest, ltest: objective and loss for trainig set
    # - history: tf fit histrory
    # Must have verbose=2 to verbose the fit

    Niter = 1  # maximum number of attempts to fit

    # Create model fit and evaluate
    for kiter in range(Niter):  # Looping until properly trained
        if "AMN_Wt" in parameter.model_type:
            # we have to recreate the object model with AMN-Wt
            model = Neural_Model(
                trainingfile=parameter.trainingfile,
                objective=parameter.objective,
                model=parameter.model,
                model_type=parameter.model_type,
                scaler=parameter.scaler,
                input_dim=parameter.input_dim,
                output_dim=parameter.output_dim,
                n_hidden=parameter.n_hidden,
                hidden_dim=parameter.hidden_dim,
                activation=parameter.activation,
                timestep=parameter.timestep,
                learn_rate=parameter.learn_rate,
                decay_rate=parameter.decay_rate,
                regression=parameter.regression,
                epochs=parameter.epochs,
                train_rate=parameter.train_rate,
                dropout=parameter.dropout,
                batch_size=parameter.batch_size,
                niter=parameter.niter,
                xfold=parameter.xfold,
                es=parameter.es,
                verbose=verbose,
            )
            model.X, model.Y = Xtrain, Ytrain
        else:
            model = parameter
        Net = model_type(model, verbose=verbose)
        # early stopping
        es = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=verbose)
        callbacks = [es] if model.es else []
        # fit
        v = True if verbose == 2 else False
        history = Net.model.fit(
            Xtrain,
            Ytrain,
            validation_data=(Xtest, Ytest),
            epochs=model.epochs,
            batch_size=model.batch_size,
            callbacks=callbacks,
            verbose=v,
        )
        # evaluate
        ytrain, stats = evaluate_model(
            Net.model, Xtrain, Ytrain, model, verbose=verbose
        )
        otrain, ltrain = stats.train_objective[0], stats.train_loss[0]
        if otrain > 0.5:
            break
        else:
            print("looping bad training iter=%d r2=%.4f" % (kiter, otrain))

    # Hopefullly fit is > 0.5 now evaluate test set
    ytest, stats = evaluate_model(Net.model, Xtest, Ytest, model, verbose=verbose)
    otest, ltest = stats.test_objective[0], stats.test_loss[0]

    print(
        "train = %.2f test = %.2f loss-train = %6f loss-test = %.6f iter=%d"
        % (otrain, otest, ltrain, ltest, kiter)
    )

    return Net, ytrain, ytest, otrain, ltrain, otest, ltest, history


def train_evaluate_model(parameter, verbose=False, only_xfold=False, keep_vin=False):
    # A standard function to create a model, fit, and Kflod cross validate
    # with early stopping
    # Kfold is performed for param.xfold test sets (if param.niter = 0)
    # otherwise only for niter test sets
    # Inptuts:
    # - all necessary parameter including
    #   parameter.model, the function used to create the model
    #   parameter.input_model, the function used to shape the model inputs
    #   parameter.X and parameter.Y, the dataset
    #   parameter.regression (boolean) if false classification
    # Outputs:
    # - the best model (highest Q2/Acc on kfold test sets)
    # - the values predicted for each fold (if param.niter = 0)
    #   or the whole set when (param.niter > 0)
    # - the mean R2/Acc on the test sets
    # - the mean constraint value on the test sets
    # Must have verbose=True to verbose the fit

    param = copy.copy(parameter)
    X, Y = model_input(param, verbose=verbose)
    param.X, param.Y = X, Y
    # Train on all data
    if param.xfold < 2:  # no cross-validation
        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = train_model(
            param, X, Y, X, Y, verbose=verbose
        )
        # Return Stats
        stats = ReturnStats(otrain, 0, ltrain, 0, otest, 0, ltest, 0)
        return Net, ytrain, stats, history

    # Cross-validation loop
    Otrain, Otest, Ltrain, Ltest, Omax, Netmax, Ypred = (
        [],
        [],
        [],
        [],
        -1.0e32,
        None,
        np.copy(Y),
    )

    kfold = KFold(n_splits=param.xfold, shuffle=True)
    kiter = 0
    all_out = []
    vin = []
    for train, test in kfold.split(X, Y):
        if verbose:
            print("-------train", train)
        if verbose:
            print("-------test ", test)
        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = train_model(
            param, X[train], Y[train], X[test], Y[test], verbose=verbose
        )
        # compile Objective (O) and Constraint (C) for train and test
        Otrain.append(otrain)
        Otest.append(otest)
        Ltrain.append(ltrain)
        Ltest.append(ltest)

        # in case y does not have the same shape than Y
        if Ypred.shape[1] != ytest.shape[1]:
            vin_part = ytest[:, 4+Net.S.shape[1]:]
            n, m, p = Y.shape[0], ytest.shape[1], vin_part.shape[1]+1
            Ypred = np.zeros(n * m).reshape(n, m)
            Vin = np.zeros(n * p).reshape(n, p)
        for i in range(len(test)):
            Ypred[test[i]] = ytest[i]
            Vin[test[i]] = np.append(ytest[i, 4+Net.S.shape[1]:], Y[test[i], 0])
        # Get the best network
        (Omax, Netmax) = (otest, Net) if otest > Omax else (Omax, Netmax)
        kiter += 1
        if (param.niter > 0 and kiter >= param.niter) or kiter >= param.xfold:
            break

        all_out.append([Y[test], ytest])

    # Prediction using best model on whole dataset
    Pred, _ = evaluate_model(Netmax.model, X, Y, param, verbose=verbose)
    Ypred = Pred if param.niter > 0 else Ypred


    # Get Stats
    stats = ReturnStats(
        np.mean(Otrain),
        np.std(Otrain),
        np.mean(Ltrain),
        np.std(Ltrain),
        np.mean(Otest),
        np.std(Otest),
        np.mean(Ltest),
        np.std(Ltest),
    )

    print("Vin shape", Vin.shape)
    if keep_vin:
        print("Keep Vin :", keep_vin)
        return Otest, all_out, Vin
    elif only_xfold: #Rely on clause order : TO FIX
        return Otest, all_out
    else:
        return Netmax, Ypred, stats, history, all_out


class Neural_Model:
    # To save, load & print all kinds of models including reservoirs
    def __init__(
        self,
        trainingfile=None,  # training set parameter file
        objective=None,
        model=None,  # the actual Keras model
        model_type="",  # the function called Dense, AMN, RC...
        scaler=False,  # X is not scaled by default
        input_dim=0,
        output_dim=0,  # model IO dimensions
        n_hidden=0,
        hidden_dim=0,  # default no hidden layer
        activation="relu",  # activation for last layer
        timestep=0,
        learn_rate=1.0,
        decay_rate=0.9,  # for GD in AMN
        # for all trainable models adam default learning rate = 1e-3
        regression=True,
        epochs=0,
        train_rate=1e-3,
        dropout=0.25,
        batch_size=5,
        niter=0,
        xfold=5,  # Cross valisation LOO does not work
        es=False,  # early stopping
        verbose=False,
    ):
        # Create empty object
        if model_type == "":
            return
        # model architecture parameters
        self.trainingfile = trainingfile
        self.model = model
        self.model_type = model_type
        self.objective = objective
        self.scaler = float(scaler)  # From bool to float
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.activation = activation
        # LP or QP parameters
        self.timestep = timestep
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        # Training parameters
        self.epochs = epochs
        self.regression = regression
        self.train_rate = train_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.niter = niter
        self.xfold = xfold
        self.es = es
        self.mediumbound = ""  # initialization
        # Get additional parameters (matrices)
        self.get_parameter(verbose=verbose)

    def get_parameter(self, verbose=False):
        # load parameter file if provided
        if self.trainingfile == None:
            return
        if not os.path.isfile(self.trainingfile + ".npz"):
            print(self.trainingfile + ".npz")
            sys.exit("parameter file not found")
        parameter = TrainingSet()
        parameter.load(self.trainingfile)
        if self.objective:
            parameter.filter_measure(measure=self.objective, verbose=verbose)
        self.Yall = parameter.Yall if self.objective else None
        self.mediumbound = parameter.mediumbound
        self.levmed = parameter.levmed
        self.valmed = parameter.valmed
        # matrices from parameter file
        self.S = parameter.S  # Stoichiometric matrix
        self.Pin = parameter.Pin  # Boundary matrix from reaction to medium
        self.Pout = parameter.Pout  # Measure matrix from reactions to measures
        self.V2M = parameter.V2M  # Reaction to metabolite matrix
        self.M2V = parameter.M2V  # Metabolite to reaction matrix
        self.X = parameter.X  # Training set X
        self.Y = parameter.Y  # Training set Y
        self.S_int = parameter.S_int
        self.S_ext = parameter.S_ext
        self.Q = parameter.Q
        self.P = parameter.P
        self.b_int = parameter.b_int
        self.b_ext = parameter.b_ext
        self.Sb = parameter.Sb
        self.c = parameter.c
        # Update input_dim and output_dim
        self.input_dim = self.input_dim if self.input_dim > 0 else parameter.X.shape[1]
        self.output_dim = (
            self.output_dim if self.output_dim > 0 else parameter.Y.shape[1]
        )

    def save(self, filename, verbose=False):
        fileparam = filename + "_param.csv"
        print(fileparam)
        filemodel = filename + "_model.h5"
        s = (
            str(self.trainingfile)
            + ","
            + str(self.model_type)
            + ","
            + str(self.objective)
            + ","
            + str(self.scaler)
            + ","
            + str(self.input_dim)
            + ","
            + str(self.output_dim)
            + ","
            + str(self.n_hidden)
            + ","
            + str(self.hidden_dim)
            + ","
            + str(self.activation)
            + ","
            + str(self.timestep)
            + ","
            + str(self.learn_rate)
            + ","
            + str(self.decay_rate)
            + ","
            + str(self.epochs)
            + ","
            + str(self.regression)
            + ","
            + str(self.train_rate)
            + ","
            + str(self.dropout)
            + ","
            + str(self.batch_size)
            + ","
            + str(self.niter)
            + ","
            + str(self.xfold)
            + ","
            + str(self.es)
        )
        with open(fileparam, "w") as h:
            # print(s, file = h)
            h.write(s)
        self.model.save(filemodel)

    def load(self, filename, verbose=False):
        fileparam = filename + "_param.csv"
        filemodel = filename + "_model.h5"
        if not os.path.isfile(fileparam):
            print(fileparam)
            sys.exit("parameter file not found")
        if not os.path.isfile(filemodel):
            print(filemodel)
            sys.exit("model file not found")
        # First read parameter file
        with open(fileparam, "r") as h:
            for line in h:
                K = line.rstrip().split(",")
        # model architecture
        self.trainingfile = str(K[0])
        self.model_type = str(K[1])
        self.objective = str(K[2])
        self.scaler = float(K[3])
        self.input_dim = int(K[4])
        self.output_dim = int(K[5])
        self.n_hidden = int(K[6])
        self.hidden_dim = int(K[7])
        self.activation = str(K[8])
        # GD parameters
        self.timestep = int(K[9])
        self.learn_rate = float(K[10])
        self.decay_rate = float(K[11])
        # Training parameters
        self.epochs = int(K[12])
        self.regression = True if K[13] == "True" else False
        self.train_rate = float(K[14])
        self.dropout = float(K[15])
        self.batch_size = int(K[16])
        self.niter = int(K[17])
        self.xfold = int(K[18])
        self.es = True if K[19] == "True" else False
        # Make objective a list
        self.objective = self.objective.replace("[", "")
        self.objective = self.objective.replace("]", "")
        self.objective = self.objective.replace("'", "")
        self.objective = self.objective.replace('"', "")
        self.objective = self.objective.split(",")
        # Get additional parameters (matrices)
        self.get_parameter(verbose=verbose)
        # Then load model
        if self.model_type == "AMN_Wt":
            self.model = load_model(
                filemodel,
                custom_objects={"RNNCell": RNNCell, "parameter": Neural_Model},
                compile=False,
            )
        else:
            self.model = load_model(filemodel, compile=False)

    def printout(self, filename=""):
        if filename != "":
            sys.stdout = open(filename, "a")
        print("training file:", self.trainingfile)
        print("model type:", self.model_type)
        print("model scaler:", self.scaler)
        print("model input dim:", self.input_dim)
        print("model output dim:", self.output_dim)
        print("model medium bound:", self.mediumbound)
        print("timestep:", self.timestep)
        if self.trainingfile:
            if os.path.isfile(self.trainingfile + ".npz"):
                print("training set size", self.X.shape, self.Y.shape)
        else:
            print("no training set provided")
        if self.n_hidden > 0:
            print("nbr hidden layer:", self.n_hidden)
            print("hidden layer size:", self.hidden_dim)
            print("activation function:", self.activation)
        if self.model_type == "AMN_QP" and self.timestep > 0:
            print("gradient learn rate:", self.learn_rate)
            print("gradient decay rate:", self.decay_rate)
        if self.epochs > 0:
            print("training epochs:", self.epochs)
            print("training regression:", self.regression)
            print("training learn rate:", self.train_rate)
            print("training dropout:", self.dropout)
            print("training batch size:", self.batch_size)
            print("training validation iter:", self.niter)
            print("training xfold:", self.xfold)
            print("training early stopping:", self.es)
        if filename != "":
            sys.stdout.close()


class RC_Model:
    # To save, load & print RC models
    def __init__(
        self,
        reservoirfile=None,  # reservoir file (a Neural_Model)
        scaler=False,
        X=[],  # X training data
        Y=[],  # Y training data
        model=None,  # the actual Keras model
        input_dim=0,
        output_dim=0,  # model IO dimensions
        # for prior network in RC model
        # default is n_hidden_prior=-1: no prior network
        n_hidden_prior=-1,
        hidden_dim_prior=-1,
        activation_prior="relu",
        # for post network in RC model
        # defaulf is n_hidden_post=-1: no post network
        n_hidden_post=-1,
        hidden_dim_post=-1,
        activation_post="linear",
        # for all trainable models adam default learning rate = 1e-3
        regression=True,
        epochs=0,
        train_rate=1e-3,
        dropout=0.25,
        batch_size=5,
        niter=0,
        xfold=5,  # cross validation
        es=False,  # early stopping
        verbose=False,
    ):
        if reservoirfile == None:
            sys.exit("must provide a reservoir file")
        if len(X) < 1 or len(Y) < 1:
            sys.exit("must provide X and Y arrays")

        # Training parameters
        self.reservoirfile = reservoirfile
        self.scaler = float(scaler)  # From bool to float
        self.X = X
        self.Y = Y
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.epochs = epochs
        self.regression = regression
        self.train_rate = train_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.niter = niter
        self.xfold = xfold
        self.es = es

        # Create reservoir
        self.res = Neural_Model()
        self.res.load(reservoirfile)
        self.res.get_parameter(verbose=verbose)

        # Get matrices for loss computation
        self.S = self.res.S  # Stoichiometric matrix
        self.Pin = self.res.Pin  # Boundary matrix from reaction to medium
        self.Pout = self.res.Pout  # Measurement matrix from reactions to measures
        self.mediumbound = self.res.mediumbound

        # Set RC model type
        if "AMN" in self.res.model_type:
            self.model_type = "RC_AMN"
        else:
            sys.exit("AMN is the only reservoir type handled with RC")
        self.prior, self.post = None, None

        # Set prior network
        if n_hidden_prior > -1:
            if self.mediumbound == "UB":
                # input is only the variable part of the medium (levmed>1)
                input_dim = self.input_dim
                for i in range(len(self.res.levmed)):
                    if self.res.levmed[i] == 1:
                        input_dim = input_dim - 1
                        # Scale X with valmed
                        self.X[:, i] = self.res.valmed[i]
                output_dim = input_dim
            else:
                input_dim, output_dim = self.input_dim, self.res.input_dim
            # Create prior network
            print(f"Neural network parameters: Input dimension {input_dim}, output_dim {output_dim}, n_hidden_prior {n_hidden_prior} hidden_dim {hidden_dim_prior}")
            self.prior = Neural_Model(
                model_type="ANN_Dense",
                input_dim=input_dim,
                output_dim=output_dim,
                n_hidden=n_hidden_prior,
                hidden_dim=hidden_dim_prior,
                activation=activation_prior,
            )

        # Set post network input_dim = output_dim
        # take as input the objective of the reservoir !!
        if n_hidden_post > -1:
            self.post = Neural_Model(
                model_type="ANN_Dense",
                input_dim=self.output_dim,
                output_dim=self.output_dim,
                n_hidden=n_hidden_post,
                hidden_dim=hidden_dim_post,
                activation=activation_post,
            )

    def printout(self, filename=""):
        if filename != "":
            sys.stdout = open(filename, "a")
        print("RC reservoir file:", self.reservoirfile)
        print("RC model type:", self.model_type)
        print("RC scaler:", self.scaler)
        print("RC model input dim:", self.input_dim)
        print("RC model output dim:", self.output_dim)
        print("RC model medium bound:", self.mediumbound)
        print("training set size", self.X.shape, self.Y.shape)
        print(
            "reservoir S, Pin, Pout matrices",
            self.S.shape,
            self.Pin.shape,
            self.Pout.shape,
        )
        if self.epochs > 0:
            print("RC training epochs:", self.epochs)
            print("RC training regression:", self.regression)
            print("RC training learn rate:", self.train_rate)
            print("RC training dropout:", self.dropout)
            print("RC training batch size:", self.batch_size)
            print("RC training validation iter:", self.niter)
            print("RC training xfold:", self.xfold)
            print("RC training early stopping:", self.es)
        if self.prior:
            print("--------prior network --------")
            self.prior.printout(filename)
        print("--------reservoir network-----")
        self.res.printout(filename)
        if self.post:
            print("--------post network ---------")
            self.post.printout(filename)
        if filename != "":
            sys.stdout.close()

class RC_Model:
    """
    Model with a prior layer, an AMN with frozen weights and posterior layer.

    Attributes
    ----------
    trained_AMN: str
        name of AMN that were previously trained
    scaler: int
        Scaling factor to multiply the input vector
    X:
        The inputs
    Y:
        Reference output
    model: keras model
        Complete model
    input_dim: int
        The number of nodes in the prelayer(s)
    n_hidden_prior: int
        The number of hidden layer in the prior network
    """

    def __init__(
        self,
        trained_AMN=None,  # reservoir file (a Neural_Model)
        scaler=False,
        X=[],  # X training data
        Y=[],  # Y training data
        model=None,  # the actual Keras model
        input_dim=0,
        output_dim=0,  # model IO dimensions
        # for prior network in RC model
        # default is n_hidden_prior=-1: no prior network
        n_hidden_prior=-1,
        hidden_dim_prior=-1,
        activation_prior="relu",
        # for post network in RC model
        # defaulf is n_hidden_post=-1: no post network
        n_hidden_post=-1,
        hidden_dim_post=-1,
        activation_post="linear",
        # for all trainable models adam default learning rate = 1e-3
        regression=True,
        epochs=0,
        train_rate=1e-3,
        dropout=0.25,
        batch_size=5,
        niter=0,
        xfold=5,  # cross validation
        es=False,  # early stopping
        verbose=False,
    ):
        if trained_AMN == None:
            sys.exit("must provide a reservoir file")
        if len(X) < 1 or len(Y) < 1:
            sys.exit("must provide X and Y arrays")

        # Training parameters
        self.reservoirfile = trained_AMN
        self.scaler = float(scaler)  # From bool to float
        self.X = X
        self.Y = Y
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.epochs = epochs
        self.regression = regression
        self.train_rate = train_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.niter = niter
        self.xfold = xfold
        self.es = es

        # Create reservoir
        self.res = Neural_Model()
        self.res.load(trained_AMN)
        self.res.get_parameter(verbose=verbose)

        # Get matrices for loss computation
        self.S = self.res.S  # Stoichiometric matrix
        self.Pin = self.res.Pin  # Boundary matrix from reaction to medium
        self.Pout = self.res.Pout  # Measurement matrix from reactions to measures
        self.mediumbound = self.res.mediumbound

        # Set RC model type
        if "AMN" in self.res.model_type:
            self.model_type = "RC_AMN"
        else:
            sys.exit("AMN is the only reservoir type handled with RC")
        self.prior, self.post = None, None

        # Set prior network
        if n_hidden_prior > -1:
            if self.mediumbound == "UB":
                # input is only the variable part of the medium (levmed>1)
                input_dim = self.input_dim
                for i in range(len(self.res.levmed)):
                    if self.res.levmed[i] == 1:
                        input_dim = input_dim - 1
                        # Scale X with valmed
                        self.X[:, i] = self.res.valmed[i]
                output_dim = input_dim
            else:
                input_dim, output_dim = self.input_dim, self.res.input_dim
            # Create prior network
            self.prior = Neural_Model(
                model_type="ANN_Dense",
                input_dim=input_dim,
                output_dim=output_dim,
                n_hidden=n_hidden_prior,
                hidden_dim=hidden_dim_prior,
                activation=activation_prior,
            )

        # Set post network input_dim = output_dim
        # take as input the objective of the reservoir !!
        if n_hidden_post > -1:
            self.post = Neural_Model(
                model_type="ANN_Dense",
                input_dim=self.output_dim,
                output_dim=self.output_dim,
                n_hidden=n_hidden_post,
                hidden_dim=hidden_dim_post,
                activation=activation_post,
            )

    def printout(self, filename=""):
        if filename != "":
            sys.stdout = open(filename, "a")
        print("RC reservoir file:", self.reservoirfile)
        print("RC model type:", self.model_type)
        print("RC scaler:", self.scaler)
        print("RC model input dim:", self.input_dim)
        print("RC model output dim:", self.output_dim)
        print("RC model medium bound:", self.mediumbound)
        print("training set size", self.X.shape, self.Y.shape)
        print(
            "reservoir S, Pin, Pout matrices",
            self.S.shape,
            self.Pin.shape,
            self.Pout.shape,
        )
        if self.epochs > 0:
            print("RC training epochs:", self.epochs)
            print("RC training regression:", self.regression)
            print("RC training learn rate:", self.train_rate)
            print("RC training dropout:", self.dropout)
            print("RC training batch size:", self.batch_size)
            print("RC training validation iter:", self.niter)
            print("RC training xfold:", self.xfold)
            print("RC training early stopping:", self.es)
        if self.prior:
            print("--------prior network --------")
            self.prior.printout(filename)
        print("--------reservoir network-----")
        self.res.printout(filename)
        if self.post:
            print("--------post network ---------")
            self.post.printout(filename)
        if filename != "":
            sys.stdout.close()