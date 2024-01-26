import pickle
from abc import abstractmethod

import numpy as np
from gurobipy import Model, GRB, quicksum, max_
from sklearn.cluster import KMeans
from time import time


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        indexes = np.random.randint(0, 2, (len(X)))
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)
        weights_2 = np.random.rand(num_features)

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        return np.stack([np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1)


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters, epsilon):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.L = n_pieces
        self.K = n_clusters
        self.epsilon = epsilon
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        np.random.seed(self.seed)
        model = Model("TwoClustersMIP")
        return model

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        self.n = X.shape[1]
        self.P = X.shape[0]
        maxs = np.ones(self.n)
        mins = np.zeros(self.n)

        def get_last_index(x, i):
            return np.floor(self.L * (x - mins[i]) / (maxs[i] - mins[i]))

        
        def get_bp(i, l):
            return mins[i] + l * (maxs[i] - mins[i]) / self.L

        # Vars
        ## Utilitary functions
        self.U = {
            (k, i, l): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="u_{}_{}_{}".format(k, i, l), ub=1)
                for k in range(self.K)
                for i in range(self.n)
                for l in range(self.L+1)
        }
        ## over-est and under-est
        self.sigmaxp = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigmaxp_{}".format(j), ub=1)
                # for k in range(self.K)
                for j in range(self.P)
        }
        self.sigmayp = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigmayp_{}".format(j), ub=1)
                # for k in range(self.K)
                for j in range(self.P)
        }

        self.sigmaxm = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigmaxm_{}".format(j), ub=1)
                # for k in range(self.K)
                for j in range(self.P)
        }
        self.sigmaym = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigmaym_{}".format(j), ub=1)
                # for k in range(self.K)
                for j in range(self.P)
        }

        self.delta1 = {
            (k, j): self.model.addVar(
                vtype=GRB.BINARY, name="delta1_{}_{}".format(k, j))
                for k in range(self.K)
                for j in range(self.P)
        } # 1 if X is preferred to Y for cluster k, 0 otherwise

        # self.delta2 = {
        #     (k, j): self.model.addVar(
        #         vtype=GRB.BINARY, name="delta2_{}_{}".format(k, j))
        #         for k in range(self.K)
        #         for j in range(self.P)
        # }


        # Constraints
        ## align preferences with delta variables
        M = 100
        uik_xij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l = get_last_index(X[j, i], i)
                    # print("x", X[j, i], "l", l)
                    bp = get_bp(i, l)
                    bp1 = get_bp(i, l+1)
                    uik_xij[k, i, j] = self.U[(k, i, l)] + ((X[j, i] - bp) / (bp1 - bp)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uik_yij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l = get_last_index(Y[j, i], i)
                    # print("x", X[j, i], "l", l)
                    bp = get_bp(i, l)
                    bp1 = get_bp(i, l+1)
                    uik_yij[k, i, j] = self.U[(k, i, l)] + ((Y[j, i] - bp) / (bp1 - bp)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uk_xj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_xj[k, j] = quicksum(uik_xij[k, i, j] for i in range(self.n))
        
        uk_yj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_yj[k, j] = quicksum(uik_yij[k, i, j] for i in range(self.n))
        

        #######
        # self.model.addConstrs(
        #     (quicksum((self.U[(k, i, get_last_index(X[j, i], i))] + ((X[j, i] - get_bp(i, get_last_index(X[j, i], i))) / (get_bp(i, get_last_index(X[j, i], i)+1) - get_bp(i, get_last_index(X[j, i], i)))) * (self.U[(k, i, get_last_index(X[j, i], i)+1)] - self.U[(k, i, get_last_index(X[j, i], i))]) for i in range(self.n))) -
        #       quicksum((self.U[(k, i, get_last_index(Y[j, i], i))] + ((Y[j, i] - get_bp(i, get_last_index(Y[j, i], i))) / (get_bp(i, get_last_index(Y[j, i], i)+1) - get_bp(i, get_last_index(Y[j, i], i)))) * (self.U[(k, i, get_last_index(Y[j, i], i)+1)] - self.U[(k, i, get_last_index(Y[j, i], i))]) for i in range(self.n))) + self.sigmaxm[(j)] - self.sigmaym[(j)] - self.sigmaxp[(j)] + self.sigmayp[(j)] - self.epsilon >= M*(1-self.delta1[(k,j)]) for j in range(self.P) for k in range(self.K))
        # )
        #######
        self.model.addConstrs(
            (uk_xj[k, j] - self.sigmaxp[j] + self.sigmaxm[j] - uk_yj[k, j] + self.sigmayp[j] - self.sigmaym[j] - self.epsilon >= -M*(1-self.delta1[(k,j)]) for j in range(self.P) for k in range(self.K))
        )

        # self.model.addConstrs(
        #     (quicksum(self.U[(k, i, get_last_index(X[j, i], i))] + ((X[j, i] - get_bp(i, get_last_index(X[j, i], i))) / (get_bp(i, get_last_index(X[j, i], i)+1) - get_bp(i, get_last_index(X[j, i], i)))) * (self.U[(k, i, get_last_index(X[j, i], i)+1)] - self.U[(k, i, get_last_index(X[j, i], i))]) for i in range(self.n)) -
        #       quicksum(self.U[(k, i, get_last_index(Y[j, i], i))] + ((Y[j, i] - get_bp(i, get_last_index(Y[j, i], i))) / (get_bp(i, get_last_index(Y[j, i], i)+1) - get_bp(i, get_last_index(Y[j, i], i)))) * (self.U[(k, i, get_last_index(Y[j, i], i)+1)] - self.U[(k, i, get_last_index(Y[j, i], i))]) for i in range(self.n)) + self.sigmaxm[(j)] - self.sigmaym[(j)] - self.sigmaxp[(j)] + self.sigmayp[(j)] >= self.epsilon for j in range(self.P) for k in range(self.K))
        # )
        # Ajoutez d'abord les variables binaires auxiliaires
        # z1 = {}
        # for k in range(self.K):
        #     z1[k] = self.model.addVar(vtype=GRB.BINARY)

        # Ajoutez ensuite les contraintes
        # for j in range(self.P):
        #     for k in range(self.K):
        #         self.model.addConstr(ux_minus_uy_kj[k][j] - M * (1 - self.delta1[(k, j)]) <= M*z1[k])
        #         self.model.addConstr(ux_minus_uy_kj[k][j] - M * (1 - self.delta1[(k, j)]) >= (1 - z1[k])*M)
        #     self.model.addConstr(quicksum(z1[k] for k in range(self.K)) >= 1)

        #######
        # self.model.addConstrs(
        #     (quicksum((self.U[(k, i, get_last_index(X[j, i], i))] + ((X[j, i] - get_bp(i, get_last_index(X[j, i], i))) / (get_bp(i, get_last_index(X[j, i], i)+1) - get_bp(i, get_last_index(X[j, i], i)))) * (self.U[(k, i, get_last_index(X[j, i], i)+1)] - self.U[(k, i, get_last_index(X[j, i], i))]) for i in range(self.n))) -
        #       quicksum((self.U[(k, i, get_last_index(Y[j, i], i))] + ((Y[j, i] - get_bp(i, get_last_index(Y[j, i], i))) / (get_bp(i, get_last_index(Y[j, i], i)+1) - get_bp(i, get_last_index(Y[j, i], i)))) * (self.U[(k, i, get_last_index(Y[j, i], i)+1)] - self.U[(k, i, get_last_index(Y[j, i], i))]) for i in range(self.n))) + self.sigmaxm[(j)] - self.sigmaym[(j)] - self.sigmaxp[(j)] + self.sigmayp[(j)] - self.epsilon <= M*self.delta1[(k,j)] - self.epsilon for j in range(self.P) for k in range(self.K))
        # )
        #######
        self.model.addConstrs(
            (uk_xj[k, j] - self.sigmaxp[j] + self.sigmaxm[j] - uk_yj[k, j] + self.sigmayp[j] - self.sigmaym[j] - self.epsilon <= M*self.delta1[(k,j)] - self.epsilon for j in range(self.P) for k in range(self.K))
        )
        # z2 = {}
        # for k in range(self.K):
        #     z2[k] = self.model.addVar(vtype=GRB.BINARY)

        # Ajoutez ensuite les contraintes
        # for j in range(self.P):
        #     for k in range(self.K):
        #         self.model.addConstr(-ux_minus_uy_kj[k][j] + M*self.delta1[(k,j)] <= M*z1[k])
        #         self.model.addConstr(-ux_minus_uy_kj[k][j] + M*self.delta1[(k,j)] >= (1 - z1[k])*M)
            # self.model.addConstr(quicksum(z1[k] for k in range(self.K)) >= 1)

        ## there exists a k so that delta2[k,j] = 1
        for j in range(self.P):
            self.model.addConstr(
                quicksum(self.delta1[(k, j)] for k in range(self.K)) >= 1
            )
            
        # for k in range(self.K):
        #     self.model.addConstr(
        #         quicksum(self.delta1[(k, j)] for j in range(self.P)) <= self.P - self.epsilon
        #     )
            # self.model.addConstr(
            #     quicksum(1 - self.delta1[(k, j)] for k in range(self.K)) >= 1
            # )
        
        ## Preference matching : $\forall j \in \{1,\dots,P\}, \exists k \in \{1,\dots,K\}, u_k(X[j]) - u_k(Y[j]) - \epsilon \geq 0$
        
        # self.model.addConstrs(
        #     (self.delta1[k,j] - self.delta2[k,j] == 0 for j in range(self.P) for k in range(self.K))
        # )


        ## Monothonicity : 
        # self.model.addConstrs(
        #     (quicksum((self.U[k, i, l] - self.U[k, i, l+1]) for l in range(self.L-1)) >= 0 for k in range(self.K) for i in range(self.n)))
        self.model.addConstrs(
            (self.U[(k, i, l+1)] - self.U[(k, i, l)]>=self.epsilon for k in range(self.K) for i in range(self.n) for l in range(self.L)))
        ### total score is one, start of each score is 0
        self.model.addConstrs(
            (self.U[(k, i, 0)] == 0 for k in range(self.K) for i in range(self.n)))
        self.model.addConstrs(
            (quicksum(self.U[(k, i, self.L)] for i in range(self.n)) == 1 for k in range(self.K)))
        
        # Objective
        self.model.setObjective(quicksum(self.sigmaxp[j] + self.sigmaxm[j] + self.sigmayp[j] + self.sigmaym[j] for j in range(self.P)), GRB.MINIMIZE)
        # self.model.setObjective(quicksum((self.sigmaxp[k, j] + self.sigmaxm[k, j] + self.sigmayp[k, j] + self.sigmaym[k, j]) for k in range(self.K) for j in range(self.P))+ quicksum(M*self.delta1[(k,j)] for k in range(self.K) for j in range(self.P)) + quicksum(M*quicksum(self.U[(k, i, get_last_index(X[j, i], i))] + (X[j, i] - get_bp(i, get_last_index(X[j, i], i))) / (get_bp(i, get_last_index(X[j, i], i)+1) - get_bp(i, get_last_index(X[j, i], i))) * (self.U[(k, i, get_last_index(X[j, i], i)+1)] - self.U[(k, i, get_last_index(X[j, i], i))]) for i in range(self.n)) -
        #       quicksum(self.U[(k, i, get_last_index(Y[j, i], i))] + (Y[j, i] - get_bp(i, get_last_index(Y[j, i], i))) / (get_bp(i, get_last_index(Y[j, i], i)+1) - get_bp(i, get_last_index(Y[j, i], i))) * (self.U[(k, i, get_last_index(Y[j, i], i)+1)] - self.U[(k, i, get_last_index(Y[j, i], i))]) for i in range(self.n)) + self.sigmaxm[(k, j)] - self.sigmaym[(k, j)] - self.sigmaxp[(k, j)] + self.sigmayp[(k, j)] - self.epsilon for k in range(self.K) for j in range(self.P)) , GRB.MINIMIZE)


        def plot_utilitary_fns(U):
            import matplotlib.pyplot as plt
            for k in range(self.K):
                for i in range(self.n):
                    plt.plot([get_bp(i, l) for l in range(self.L+1)], [U[k, i, l] for l in range(self.L+1)])
                plt.legend(["feature {}".format(i) for i in range(self.n)])
                plt.show()
        # Solve
        self.model.params.outputflag = 0  # mode muet
        self.model.update()
        self.model.optimize()
        if self.model.status == GRB.INFEASIBLE:
            print("\n le PROGRAMME N'A PAS DE SOLUTION!!!")
            raise Exception("Infeasible")
        elif self.model.status == GRB.UNBOUNDED:
            print("\n le PROGRAMME EST NON BORNÉ!!!")
            raise Exception("Unbounded")
        else:
            print("\n le PROGRAMME A UNE SOLUTION!!!")
            # print the value of objective function
            print("objective function value: ", self.model.objVal)
            self.U = {(k, i, l): self.U[k, i, l].x for k in range(self.K) for i in range(self.n) for l in range(self.L+1)}
            # self.sigmaxp = {(j): self.sigmaxp[(j)].x for j in range(self.P)}
            # self.sigmayp = {(j): self.sigmayp[(j)].x for j in range(self.P)}
            # self.sigmaxm = {(j): self.sigmaxm[(j)].x for j in range(self.P)}
            # self.sigmaym = {(j): self.sigmaym[(j)].x for j in range(self.P)}
            # print(self.sigmaxm)
            # print(self.sigmaym)
            self.delta1 = {(k, j): self.delta1[k, j].x for k in range(self.K) for j in range(self.P)}


            
            plot_utilitary_fns(self.U)
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        maxs = np.ones(self.n)
        mins = np.zeros(self.n)

        def get_bp(i, l):
            segments = np.linspace(mins[i], maxs[i], self.L + 1)
            if l >= len(segments):
                return segments[-1]
            return segments[l]

        def get_last_index(x, i):
            segments = np.linspace(mins[i], maxs[i], self.L + 1)
            last_index = np.argmax(x < segments) - 1
            return last_index
        
        utilities = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            for j in range(X.shape[0]):
                for i in range(self.n):
                    l = get_last_index(X[j, i], i)
                    utilities[j, k] += self.U[k, i, get_last_index(X[j, i], i)] + ((X[j, i] - get_bp(i, get_last_index(X[j, i], i))) / (get_bp(i, get_last_index(X[j, i], i)+1) - get_bp(i, get_last_index(X[j, i], i)))) * (self.U[k, i, get_last_index(X[j, i], i)+1] - self.U[k, i, get_last_index(X[j, i], i)])
        # add sigmas
        # for k in range(self.K):
        #     for j in range(X.shape[0]):
        #         utilities[j, k] += self.sigmaxp[k, j] + self.sigmaxm[k, j] - self.sigmayp[k, j] - self.sigmaym[k, j]
        return utilities


class HeuristicModel(BaseModel):
    """Skeleton of Heuristic you have to write as the second exercise.
    Heuristic is a model that does not use MIP but rather a heuristic to find the best cluster for each element.
    Heuristic : Genetic algorithm :
        - Individual : utilitary function for each cluster
        - Fitness : percentage of pairs explained by at least a cluster
        - Selection : tournament selection
        - Crossover : one point crossover
        - Mutation : random mutation
    """

    def __init__(self, n_pieces, n_clusters, epsilon):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.L = n_pieces
        self.K = n_clusters
        self.epsilon = epsilon
        # self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        np.random.seed(self.seed)
        models = [Model(f"HeuristicModelCluster{k}") for k in range(self.K)]
        return models

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        np.random.seed(self.seed)
        self.n = X.shape[1]
        self.P = X.shape[0]
        maxs = np.ones(self.n)
        mins = np.zeros(self.n)

        def get_last_index(x, i):
            return int(np.floor(self.L * (x - mins[i]) / (maxs[i] - mins[i])))
        
        def get_bp(i, l):
            return mins[i] + l * (maxs[i] - mins[i]) / self.L
        
        def get_utility(x, U, k, i, l):
            return U[k, i, l] + ((x - get_bp(i, l)) / (get_bp(i, l+1) - get_bp(i, l))) * (U[k, i, l+1] - U[k, i, l])
        
        def score_U(U, X, Y):
            Ux = np.zeros((self.P, self.K))
            Uy = np.zeros((self.P, self.K))
            for j in range(self.P):
                x, y = X[j], Y[j]
                for k in range(self.K):
                    for i in range(self.n):
                        lx = get_last_index(x[i], i)
                        Ux[j, k] += get_utility(x[i], U, k, i, lx)
                        ly = get_last_index(y[i], i)
                        Uy[j, k] += get_utility(y[i], U, k, i, ly)
            return np.sum(np.sum(Ux - Uy > 0, axis=1) > 0) / len(Ux)
        
        def constrain_U(U):
            """
            Modify U so that it respects the constraints.
            U (K, n, L+1) should be :
                - monotonous, meaning that U[k, i, l] <= U[k, i, l+1] for all k, i, l
                - sum to 1, meaning that sum(U[k, :, L]) = 1 for all k, i
                - start at 0, meaning that U[k, i, 0] = 0 for all k, i
            """
            for k in range(self.K):
                for i in range(self.n):
                    U[k, i, :] = np.sort(U[k, i, :])
            for k in range(self.K):
                # all last values accross each feature must sum to 1
                U[k, :, :] = U[k, :, :] / np.sum(U[k, :, -1])
            return U
        
        def verify_U(U):
            """
            Verify that U respects the constraints.
            """
            for k in range(self.K):
                for i in range(self.n):
                    if not np.all(U[k, i, :] == np.sort(U[k, i, :])):
                        return False, f"not sorted: {U[k, i, :]}"
                    if not np.allclose(U[k, i, 0], 0):
                        return False, f"not start at 0: {U[k, i, 0]}"
                # if the sum of the last values is not between 0.99 and 1.01, we have a problem
                if not np.allclose(np.sum(U[k, :, -1]), 1):
                    return False, f"not sum to 1: {np.sum(U[k, :, -1])}"
            return True, ""
        
        def crossover(parent1, parent2):
            """
            We mix a random number of feature utilities from parent1 and the rest from parent2
            """
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
            for k in range(self.K):
                n_i1 = np.random.randint(0, self.n)
                n_i2 = np.random.randint(0, self.n)
                i1s = np.unique(np.random.randint(0, self.n, n_i1))
                i2s = np.unique(np.random.randint(0, self.n, n_i2))
                for i in i1s:
                    child1[k, i, :] = parent2[k, i, :]
                for i in i2s:
                    child2[k, i, :] = parent1[k, i, :]
            child1, child2 = constrain_U(child1), constrain_U(child2)
            return child1, child2
        
        def mutation(child):
            """
            We change the value of a random number of feature utilities
            """
            for k in range(self.K):
                n_i = np.random.randint(0, self.n)
                i_s = np.unique(np.random.randint(0, self.n, n_i))
                for i in i_s:
                    n_l = np.random.randint(0, self.L//2)
                    l_s = np.unique(np.random.randint(1, self.L+1, n_l))
                    for l in l_s:
                        child[k, i, l] = np.random.rand()
                # child[k, :, :] = np.sort(child[k, :, :], axis=-1)
                # # all last values accross each feature must sum to 1
                # child[k, :, -1] = 1 - np.sum(child[k, :, :-1], axis=1)
            child = constrain_U(child)
            return child


        pop_size = 50
        n_epochs = 500

        # Init
        self.Us = np.zeros((pop_size, self.K, self.n, self.L+1))
        self.Us[:, :, :, -1] = np.random.rand(pop_size, self.K, self.n)
        self.Us[:, :, :, -1] = self.Us[:, :, :, -1]/np.sum(self.Us[:, :, :, -1], axis=2)[:, :, np.newaxis]
        for l in range(1, self.L):
            # all the other values are random, comprised between 0 and the last value (self.Us[:, :, :, -1])
            self.Us[:, :, :, l] = np.random.rand(pop_size, self.K, self.n) * self.Us[:, :, :, -1]
        self.Us[:, :, :, :] = np.sort(self.Us[:, :, :, :], axis=-1)
        
        # print(self.Us)
        best_U = np.ones((self.K, self.n, self.L+1))
        best_score = 0

        

        for e in range(n_epochs):
            start_t = time()
            new_Us = np.zeros((pop_size, self.K, self.n, self.L+1))
            scores = np.array([score_U(self.Us[i], X, Y) for i in range(pop_size)])
            # Ranking
            ranking = np.argsort(scores)
            print(f"Best score at iteration {e} : {scores[ranking[-1]]}")
            if scores[ranking[-1]] > best_score:
                best_score = scores[ranking[-1]]
                best_U = self.Us[ranking[-1]]

            # Selection
            parents = ranking[int(pop_size/2):].tolist()
            new_Us[int(pop_size/2):] = self.Us[ranking[int(pop_size/2):]]
            # print(parents)

            # Crossover
            for i in range(0, int(pop_size/2), 2):
                parent1 = np.random.choice(parents)
                parent2 = np.random.choice(parents)
                child1, child2 = crossover(self.Us[parent1], self.Us[parent2])
                new_Us[i] = child1
                new_Us[i+1] = child2
            
            
            
            # Mutation
            n_mutations = np.random.randint(0, pop_size)
            population = np.arange(pop_size).tolist()
            for i in range(n_mutations):
                mutated = population.pop(np.random.randint(0, len(population)))
                new_Us[mutated] = mutation(new_Us[mutated])


            for U in new_Us:
                isok, msg = verify_U(U)
                assert isok, msg
            self.Us = new_Us
            print(f"\tEpoch {e} took {round(time() - start_t)} seconds")
            print(f"\tEstimated time remaining : {round((n_epochs - e) * (time() - start_t))} seconds")
        
        
        scores = np.array([score_U(self.Us[i], X, Y) for i in range(pop_size)])
        # Ranking
        ranking = np.argsort(scores)

        if scores[ranking[-1]] > best_score:
            best_score = scores[ranking[-1]]
            best_U = self.Us[ranking[-1]]
        
        self.U = best_U
        np.save("best_U.npy", self.U)
        print(f"Best score : {best_score}")

        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        maxs = np.ones(self.n)
        mins = np.zeros(self.n)

        def get_bp(i, l):
            segments = np.linspace(mins[i], maxs[i], self.L + 1)
            if l >= len(segments):
                return segments[-1]
            return segments[l]

        def get_last_index(x, i):
            segments = np.linspace(mins[i], maxs[i], self.L + 1)
            last_index = np.argmax(x < segments) - 1
            return last_index
        
        utilities = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            for j in range(X.shape[0]):
                for i in range(self.n):
                    l = get_last_index(X[j, i], i)
                    utilities[j, k] += self.U[k, i, get_last_index(X[j, i], i)] + ((X[j, i] - get_bp(i, get_last_index(X[j, i], i))) / (get_bp(i, get_last_index(X[j, i], i)+1) - get_bp(i, get_last_index(X[j, i], i)))) * (self.U[k, i, get_last_index(X[j, i], i)+1] - self.U[k, i, get_last_index(X[j, i], i)])
        # add sigmas
        # for k in range(self.K):
        #     for j in range(X.shape[0]):
        #         utilities[j, k] += self.sigmaxp[k, j] + self.sigmaxm[k, j] - self.sigmayp[k, j] - self.sigmaym[k, j]
        return utilities


if __name__ == "__main__":
    import sys

    sys.path.append("../python/")

    import matplotlib.pyplot as plt
    import numpy as np
    from data import Dataloader
    from models import RandomExampleModel
    import metrics
    # Loading the data
    data_loader = Dataloader("../data/dataset_4") # Specify path to the dataset you want to load
    X, Y = data_loader.load()
    
    parameters = {"n_pieces": 5,
              "n_clusters" : 2,
              "epsilon" : 0.00001} # Can be completed
    model = HeuristicModel(**parameters)
    model.fit(X, Y)