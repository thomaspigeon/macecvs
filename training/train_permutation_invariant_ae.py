import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler



class TrainAE:
    """Class to train AE models

    The dataset is the described in the __init__. It is first split into a training dataset and a test dataset. This
    last one is not used at all for validation of hyperparameters. It is used to computed conditional averages and
    other quantities to charaterize the convergence of the training. The training dataset has to be split into K  folds
    to generate the validation and test data.
    """

    def __init__(self, ae, dataset, standardize=False, zca_whiten=False):
        """

        :param ae:                  AE model from autoencoders.ae_models.DeepAutoEncoder
        :param dataset:             dict, with dataset["boltz_points"] a np.array with ndim==2, shape==[any, 2] an array
                                    of points on the 2D potentials distributed according ot the boltzmann gibbs measure.
                                    Optionally,  dataset["boltz_weights"] is a np.array with ndim==2, shape==[any, 1],
                                    if not provided the weights are set to 1. Another option is dataset["react_points"],
                                    np.array with ndim==2, shape==[any, 2] an array  of points on the 2D potentials
                                    distributed according ot the probability measure of reactive trajectories.
                                    dataset["react_weights"] can be set as well, set to 1 if not provided
        :param penalization_points: np.array, ndim==2, shape=[any, 3], penalization_point[:, :2] are the points on which
                                    the encoder is penalized if its values on these points differ from
                                    penalization_point[:, 2]
        :param standardize:         boolean, whether the points should be rescaled so that the average in every
                                    direction is zero and has variance equal to 1  for the "boltz_points" WARNING the
                                    weights are currently not considered in this operation
        :param zca_whiten:          boolean, whether the data should be whitened using mahalanobis whitening fitted on
                                    the "boltz_points" WARNING the weights are currently not considered in this
                                    operation
        """
        self.ae = ae
        self.dataset = dataset
        self.standardize = standardize
        self.zca_whiten = zca_whiten
        if standardize:
            self.scaler = StandardScaler()
            self.dataset["boltz_points"] = self.scaler.fit_transform(dataset["boltz_points"])
            if "react_points" in dataset.keys():
                self.dataset["react_points"] = self.scaler.transform(dataset["react_points"])
        elif zca_whiten:
            cov_matrix = np.cov(dataset["boltz_points"], rowvar=False)  # Compute covariance matrix
            U, D, V = np.linalg.svd(cov_matrix)  # Single value decompostion
            epsilon = 1e-12  # Small value to prevent division by 0
            self.ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(D + epsilon)), U.T))
            self.dataset["boltz_points"] = self.ZCAMatrix.dot(dataset["boltz_points"].T).T
            if "react_points" in dataset.keys():
                self.dataset["react_points"] = self.ZCAMatrix.dot(dataset["react_points"].T).T
        self.training_dataset = None
        self.test_dataset = None
        self.Kfold_splits = None
        self.train_data = None
        self.validation_data = None
        self.mse_weight = None
        self.l1_pen_weight = None
        self.l2_pen_weight = None
        self.pen_points_weight = None
        self.pen_points_recons = None
        self.n_wait = None
        self.contractive_weight = None

    def set_loss_weight(self, loss_params):
        """Function to set the loss parameters.

        :param loss_params:     dict, containing: loss_params["mse_weight"] float >= 0, prefactor of the MSE term
                                of the loss, loss_params["contractive_weight"], float >= 0, prefactor of the
                                squared gradient the encoder,
                                loss_params["l1_pen_weight"], float >= 0, prefactor of the L1 weight decay penalization,
                                loss_params["l2_pen_weight"], float >= 0, prefactor of the L2 weight decay penalization,
                                loss_params["pen_points_weight"], float >= 0, prefactor of the penalization so that
                                certain points have a certain encoded value.
                                loss_params["pen_points_recons"], float >= 0, prefactor of the penalization so that
                                certain points with given encoded values have small reconstruction loss.
                                loss_params["n_wait"], int >= 1, early stopping parameter. If the test loss has not
                                decreased for n_wait epochs, the training is stopped and the model kept in self.ae is
                                the one corresponding to the minimal test loss
        """
        if "mse_weight" not in loss_params.keys():
            raise ValueError("""loss_params["mse_weight"] must be set as a float >= 0.""")
        elif type(loss_params["mse_weight"]) != float or loss_params["mse_weight"] < 0.:
            raise ValueError("""loss_params["mse_weight"] must be set as a float >= 0.""")
        else:
            self.mse_weight = loss_params["mse_weight"]


        if "pen_points_weight" not in loss_params.keys():
            self.pen_points_weight = 0.
            print("""pen_points_weight value not provided, set to default value of: """, self.pen_points_weight)
        elif type(loss_params["pen_points_weight"]) != float or loss_params["pen_points_weight"] < 0.:
            raise ValueError("""loss_params["pen_points_weight"] must be set as a float >= 0.""")
        else:
            self.pen_points_weight = loss_params["pen_points_weight"]

        if "pen_points_recons" not in loss_params.keys():
            self.pen_points_recons = 0.
            print("""pen_points_recons value not provided, set to default value of: """, self.pen_points_recons)
        elif type(loss_params["pen_points_recons"]) != float or loss_params["pen_points_recons"] < 0.:
            raise ValueError("""loss_params["pen_points_recons"] must be set as a float >= 0.""")
        else:
            self.pen_points_recons = loss_params["pen_points_recons"]

        if "l1_pen_weight" not in loss_params.keys():
            self.l1_pen_weight = 0
            print("""l1_pen_weight value not provided, set to default value of: """, self.l1_pen_weight)
        elif type(loss_params["l1_pen_weight"]) != float or loss_params["l1_pen_weight"] < 0.:
            raise ValueError("""loss_params["l1_pen_weight"] must be a float >= 0.""")
        else:
            self.l1_pen_weight = loss_params["l1_pen_weight"]

        if "l2_pen_weight" not in loss_params.keys():
            self.l2_pen_weight = 0
            print("""l2_pen_weight value not provided, set to default value of: """, self.l2_pen_weight)
        elif type(loss_params["l2_pen_weight"]) != float or loss_params["l2_pen_weight"] < 0.:
            raise ValueError("""loss_params["l2_pen_weight"] must be a float >= 0.""")
        else:
            self.l2_pen_weight = loss_params["l2_pen_weight"]

        if "n_wait" not in loss_params.keys():
            self.n_wait = 10
            print("""n_wait value not provided, set to default value of: """, self.n_wait)
        elif type(loss_params["n_wait"]) != int or loss_params["n_wait"] < 1:
            raise ValueError("""loss_params["n_wait"] must be a int >= 1""")
        else:
            self.n_wait = loss_params["n_wait"]

        # --- new: contractive weight ---
        if "contractive_weight" not in loss_params.keys():
            self.contractive_weight = 0.0
            print("""contractive_weight not provided, set to default value of: """, self.contractive_weight)
        elif type(loss_params["contractive_weight"]) != float or loss_params["contractive_weight"] < 0.:
            raise ValueError("""loss_params["contractive_weight"] must be a float >= 0.""")
        else:
            self.contractive_weight = loss_params["contractive_weight"]

    def set_dataset(self, dataset):
        """Method to reset dataset

        :param dataset:             dict, with dataset["boltz_points"] a np.array with ndim==2, shape==[any, 2] an array
                                    of points on the 2D potentials distributed according ot the boltzmann gibbs measure.
                                    Optionally,  dataset["boltz_weights"] is a np.array with ndim==2, shape==[any, 1],
                                    if not provided the weights are set to 1. Another option is dataset["react_points"],
                                    np.array with ndim==2, shape==[any, 2] an array  of points on the 2D potentials
                                    distributed according ot the probability measure of reactive trajectories.
                                    dataset["react_weights"] can be set as well, set to 1 if not provided
        """
        self.dataset = dataset
        self.training_dataset = None
        self.test_dataset = None
        self.Kfold_splits = None
        self.train_data = None
        self.validation_data = None

    def train_test_split(self, train_size=None, test_size=None, seed=None):
        """Method to separate the dataset into training and test dataset.

        :param train_size:  float or int, if float represents the proportion to include in the train split. If int, it
                            corresponds to the exact number of train samples. If None, it is set to be the complement of
                            the test_size. If both are None, it is set to 0.75
        :param test_size:   float or int, if float represents the proportion to include in the test split. If int, it
                            corresponds to the exact number of test samples. If None, it is set to be the complement of
                            the train_size. If both are None, it is set to 0.25
                            corresponds to the exact number of train samples
        :param seed:        int, random state for the splitting
        """
        dset = []
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        for i in range(len(self.dataset[next(iter(self.dataset.keys()))])):
            dset.append(
                {key: torch.tensor(self.dataset[key][i].astype("float32")).to(device) for key in self.dataset.keys()}
            )
        self.training_dataset, self.test_dataset = ttsplit(dset,
                                                           test_size=test_size,
                                                           train_size=train_size,
                                                           random_state=seed)

    def split_training_dataset_K_folds(self, n_splits, seed=None):
        """ Allows to split the training dataset into multiple groups to optimize eventual hyperparameter

        :param n_splits: int, number of splits, must be int >= 2
        :param seed:     int, random state
        """
        if n_splits < 2:
            raise ValueError("The number of splits must be superior or equal to 2")
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        self.Kfold_splits = []
        for i, fold in kf.split(self.training_dataset):
            self.Kfold_splits.append(fold)

    def set_train_val_data(self, split_index):
        """Set the training and validation set

        :param split_index:    int, the split of the training data_set, should be such that 0 <= split_index <= n_splits
        """
        if split_index < 0:
            raise ValueError("The split index must be between 0 and the number of splits - 1")
        validation = []
        for i in self.Kfold_splits[split_index]:
            validation.append(self.training_dataset[i])
        indices = np.setdiff1d(range(len(self.Kfold_splits)), split_index)
        train = []
        for i in self.Kfold_splits[indices[0]]:
            train.append(self.training_dataset[i])
        if len(self.Kfold_splits) > 2:
            for i in range(1, len(indices)):
                for j in self.Kfold_splits[indices[i]]:
                    train.append(self.training_dataset[j])
        self.train_data = train
        self.validation_data = validation

    @staticmethod
    def l1_penalization(model):
        """

        :param model:       ae model
        :return l1_pen:     torch float
        """
        return sum(p.abs().sum() for p in model.parameters()) / sum(torch.numel(p) for p in model.parameters())

    @staticmethod
    def l2_penalization(model):
        """

        :param model:       ae model
        :return l1_pen:     torch float
        """
        return sum(p.pow(2.0).sum() for p in model.parameters()) / sum(torch.numel(p) for p in model.parameters())


class TrainLineInvariantAEMultipleDecoders(TrainAE):
    """Class to train AE models with multiple decoders

     The dataset is the described in the __init__. It is first split into a training dataset and a test dataset. This
     last one is not used at all for validation of hyperparameters. It is used to computed conditional averages and
     other quantities to charaterize the convergence of the training. The training dataset has to be split into K  folds
     to generate the validation and test data.
     """

    def __init__(self, ae, dataset, l_multidec=1, standardize=False, zca_whiten=False):
        """

        :param ae:                  AE model from autoencoders.ae_models.DeepAutoEncoder
        :param dataset:             dict, with dataset["points"] a np.array with ndim==3, shape==[any,N_lines,N_cols]
                                    Optionally,  dataset["weights"] is a np.array with ndim==2, shape==[any, 1],
                                    if not provided the weights are set to 1.
                                    dataset["pen_points"] a np.array with ndim==3, shape==[any,N_lines,N_col], points on
                                    which the encoded features should be equal to a given value
                                    dataset["pen_points_values"] a np.array with ndim==3, shape==[any,N_bottleneck], the
                                    corresponding values
        :param standardize:         boolean, whether the points should be rescaled so that the average in every
                                    direction is zero and has variance equal to 1
        :param zca_whiten:          boolean, whether the data should be whitened using mahalanobis whitening
        """
        super().__init__(ae,
                         dataset,
                         standardize=standardize,
                         zca_whiten=zca_whiten)
        self.optimizer = None
        self.l_multidec = l_multidec
        self.softmin_multidec = torch.nn.Softmin(dim=0)

    def set_optimizer(self, opt, learning_rate, parameters_to_train='all'):
        """

        :param opt:                 str, 'Adam' only.
        :param learning_rate:       float, value of the learning rate, typically 10**(-3) or smaller gives good results
                                    on the tested potentials
        :param parameters_to_train: str, either 'encoder', 'decoders',  or 'all' to set what are the trained parameters
        """
        if opt == 'Adam' and parameters_to_train == 'all':
            self.optimizer = torch.optim.Adam(
                [{'params': self.ae.parameters()}],
                lr=learning_rate)

        elif opt == 'Adam' and parameters_to_train == 'encoder':
            self.optimizer = torch.optim.Adam(
                [{'params': self.ae.line_encoder.parameters()}] +
                [{'params': self.ae.column_encoder.parameters()}], lr=learning_rate)

        elif opt == 'Adam' and parameters_to_train == 'decoders':
            self.optimizer = torch.optim.Adam(
                [{'params': self.ae.line_decoders.parameters()}] +
                [{'params': self.ae.column_decoders.parameters()}],
                lr=learning_rate)
        else:
            raise ValueError("""The parameters opt and parameters_to_train must be specific str, see docstring""")

    def contractive_loss(self, batch, lam=1e-3):
        """
        Contractive loss: penalizes the sensitivity of the encoder to input perturbations.

        :param batch: dict, containing at least "points"
        :param lam: float, weight of the contractive penalty
        :return: torch scalar
        """
        x = batch["points"]
        x.requires_grad_(True)  # Enable gradient wrt inputs

        # Encode input
        z = self.ae.encoded(x)

        # We want ||∂z/∂x||^2
        # Compute gradient of each latent dimension wrt inputs
        contractive_penalty = 0.0
        for i in range(z.shape[1]):  # iterate over latent dims
            grad = torch.autograd.grad(
                outputs=z[:, i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            contractive_penalty += torch.sum(grad.pow(2))

        # Normalize by batch size
        contractive_penalty = contractive_penalty / x.shape[0]

        return lam * contractive_penalty

    def mse_loss(self, batch):
        """MSE term.

        :param batch:   batch dict with at leat the keys: "points","weights",
        :return mse:    torch float, mean squared error between input and output for points distributed according to
                        Boltzmann-Gibbs measure for each decoders
        """
        errors = [self.ae.decoded(batch["points"], i) for i in range(len(self.ae.column_decoders))]
        errors = [
            torch.sum((batch["points"].reshape(len(batch["points"]), self.ae.n_lines, 1, self.ae.n_columns)
                       - error.reshape(len(batch["points"]), 1, self.ae.n_lines, self.ae.n_columns)) ** 2, dim=-1)
            for error in errors
        ]
        for j, error in enumerate(errors):
            _, posi = torch.min(error, dim=-1, keepdim=True)
            mask = torch.zeros_like(error).scatter(-1, posi, 1.)
            previous = torch.zeros_like(mask[:, 0:1, :])
            for i in range(1, self.ae.n_lines):
                previous += mask[:, i - 1:i, :] * 10 ** 15
                _, posi = torch.min(previous + error, dim=-1, keepdim=True)
                mask[:, i:i + 1, :] = torch.zeros_like(error).scatter(-1, posi, 1.)[:, i:i + 1, :]
            errors[j] = (mask * error).sum(dim=[-1, -2])
        errors = torch.stack([batch["weights"] * error for error in errors])
        errors = (self.softmin_multidec(self.l_multidec * errors) * errors).sum(dim=0)
        return torch.mean(errors)

    def local_penalization_loss_enc(self, batch):
        enc = self.ae.encoded(batch["pen_points"])
        err = torch.nn.functional.smooth_l1_loss(enc, batch["pen_points_values"])
        return torch.mean(err)

    def local_penalization_loss_dec(self, batch):
        errors = [self.ae.decoded(batch["pen_points"], i) for i in range(len(self.ae.column_decoders))]
        errors = [
            torch.sum((batch["pen_points"].reshape(len(batch["pen_points"]), self.ae.n_lines, 1, self.ae.n_columns)
                       - error.reshape(len(batch["pen_points"]), 1, self.ae.n_lines, self.ae.n_columns)) ** 2, dim=-1)
            for error in errors
        ]
        for j, error in enumerate(errors):
            _, posi = torch.min(error, dim=-1, keepdim=True)
            mask = torch.zeros_like(error).scatter(-1, posi, 1.)
            previous = torch.zeros_like(mask[:, 0:1, :])
            for i in range(1, self.ae.n_lines):
                previous += mask[:, i - 1:i, :] * 10 ** 15
                _, posi = torch.min(previous + error, dim=-1, keepdim=True)
                mask[:, i:i + 1, :] = torch.zeros_like(error).scatter(-1, posi, 1.)[:, i:i + 1, :]
            errors[j] = (mask * error).sum(dim=[-1, -2])
        errors = torch.stack(errors)
        return torch.mean(errors)

    def train(self, batch_size, max_epochs):
        """ Do the training of the model self.ae

        :param batch_size:      int >= 1, batch size for the mini-batching
        :param max_epochs:      int >= 1, maximal number of epoch of training
        :return loss_dict:      dict, contains the average loss for each epoch and its various components.
        """
        if self.optimizer is None:
            print("""The optimizer has not been set, see set_optimizer method. It is set to use 'Adam' optimizer \n 
                     with a 0.001 learning rate and optimize all the parameters of the model""")
            self.set_optimizer('Adam', 0.001)
        # move model to device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.ae.to(device)
        # prepare the various loss list to store
        loss_dict = {"train_loss": [], "test_loss": [], "train_mse": [], "test_mse": []}
        if self.contractive_weight > 0:
            loss_dict["train_contractive"] = []
            loss_dict["test_contractive"] = []
        if 'pen_points' in self.dataset.keys():
            loss_dict["train_pen_points"] = []
            loss_dict["test_pen_points"] = []
            loss_dict["train_pen_points_recons"] = []
            loss_dict["test_pen_points_recons"] = []
        train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(dataset=self.validation_data, batch_size=batch_size, shuffle=False)
        epoch = 0
        model = copy.deepcopy(self.ae)
        while epoch < max_epochs:
            loss_dict["train_loss"].append([])
            loss_dict["train_mse"].append([])
            if self.contractive_weight > 0:
                loss_dict["train_contractive"].append([])
            if "pen_points" in self.dataset.keys():
                loss_dict["train_pen_points"].append([])
                loss_dict["train_pen_points_recons"].append([])
            # train mode
            self.ae.train()
            for iteration, batch in enumerate(train_loader):
                # Set gradient calculation capabilities
                for key in batch.keys():
                    batch[key].requires_grad_()
                # Set the gradient of with respect to parameters to zero
                self.optimizer.zero_grad()
                # Compute the various loss terms
                mse = self.mse_loss(batch)
                l1_pen = self.l1_penalization(self.ae)
                l2_pen = self.l2_penalization(self.ae)
                loss = self.mse_weight * mse + \
                       self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen
                if "pen_points" in self.dataset.keys():
                    pen_points = self.local_penalization_loss_enc(batch)
                    loss += self.pen_points_weight * pen_points
                    recons_points = self.local_penalization_loss_dec(batch)
                    loss += self.pen_points_recons * recons_points
                    loss_dict["train_pen_points"][epoch].append(pen_points.to('cpu').detach().numpy())
                    loss_dict["train_pen_points_recons"][epoch].append(recons_points.to('cpu').detach().numpy())
                if self.contractive_weight > 0:
                    contractive = self.contractive_weight * self.contractive_loss(batch, lam=1)
                    loss += contractive
                    loss_dict["train_contractive"][epoch].append(contractive.to('cpu').detach().numpy())
                loss.backward()
                self.optimizer.step()
                loss_dict["train_loss"][epoch].append(loss.to('cpu').detach().numpy())
                loss_dict["train_mse"][epoch].append(mse.to('cpu').detach().numpy())

            loss_dict["train_loss"][epoch] = np.mean(loss_dict["train_loss"][epoch])
            loss_dict["train_mse"][epoch] = np.mean(loss_dict["train_mse"][epoch])
            if self.contractive_weight > 0:
                loss_dict["train_contractive"][epoch] = np.mean(loss_dict["train_contractive"][epoch])
            if "pen_points" in self.dataset.keys():
                loss_dict["train_pen_points"][epoch] = np.mean(loss_dict["train_pen_points"][epoch])
                loss_dict["train_pen_points_recons"][epoch] = np.mean(loss_dict["train_pen_points_recons"][epoch])
            loss_dict["test_loss"].append([])
            loss_dict["test_mse"].append([])
            if self.contractive_weight > 0:
                loss_dict["test_contractive"].append([])
            if "pen_points" in self.dataset.keys():
                loss_dict["test_pen_points"].append([])
                loss_dict["test_pen_points_recons"].append([])
            # test mode
            for iteration, batch in enumerate(valid_loader):
                # Set gradient calculation capabilities
                for key in batch.keys():
                    batch[key].requires_grad_()
                # Compute the various loss terms
                mse = self.mse_loss(batch)
                l1_pen = self.l1_penalization(self.ae)
                l2_pen = self.l2_penalization(self.ae)
                loss = self.mse_weight * mse + \
                       self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen
                if "pen_points" in self.dataset.keys():
                    pen_points = self.local_penalization_loss_enc(batch)
                    loss += self.pen_points_weight * pen_points
                    recons_points = self.local_penalization_loss_dec(batch)
                    loss += self.pen_points_recons * recons_points
                    loss_dict["test_pen_points"][epoch].append(pen_points.to('cpu').detach().numpy())
                    loss_dict["test_pen_points_recons"][epoch].append(recons_points.to('cpu').detach().numpy())
                if self.contractive_weight > 0:
                    contractive = self.contractive_weight * self.contractive_loss(batch, lam=1)
                    loss += contractive
                    loss_dict["test_contractive"][epoch].append(contractive.to('cpu').detach().numpy())
                loss_dict["test_loss"][epoch].append(loss.to('cpu').detach().numpy())
                loss_dict["test_mse"][epoch].append(mse.to('cpu').detach().numpy())

            loss_dict["test_loss"][epoch] = np.mean(loss_dict["test_loss"][epoch])
            loss_dict["test_mse"][epoch] = np.mean(loss_dict["test_mse"][epoch])
            if self.contractive_weight > 0:
                loss_dict["test_contractive"][epoch] = np.mean(loss_dict["test_contractive"][epoch])
            if "pen_points" in self.dataset.keys():
                loss_dict["test_pen_points"][epoch] = np.mean(loss_dict["test_pen_points"][epoch])
                loss_dict["test_pen_points_recons"][epoch] = np.mean(loss_dict["test_pen_points_recons"][epoch])

            # Early stopping
            if loss_dict["test_loss"][epoch] == np.min(loss_dict["test_loss"]):
                model = copy.deepcopy(self.ae)
            if epoch >= self.n_wait:
                if np.min(loss_dict["test_loss"]) < np.min(loss_dict["test_loss"][- self.n_wait:]):
                    epoch = max_epochs
                    self.ae = model
            epoch += 1
        print("training ends after " + str(len(loss_dict["test_loss"])) + " epochs.\n")
        return loss_dict

    def print_test_loss(self, batch_size):
        """Print the test loss and its various components"""
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)
        results = {"loss": [],
                   "mse": []}
        if "pen_points" in self.dataset.keys():
            results[("pen_points")] = []
            results[("pen_points_recons")] = []
        if self.contractive_weight > 0:
            results[("contractive")] = []
        for iteration, batch in enumerate(test_loader):
            # Set gradient calculation capabilities
            for key in batch.keys():
                batch[key].requires_grad_()

            mse = self.mse_loss(batch)
            l1_pen = self.l1_penalization(self.ae)
            l2_pen = self.l2_penalization(self.ae)
            loss = self.mse_weight * mse + \
                   self.l1_pen_weight * l1_pen + \
                   self.l2_pen_weight * l2_pen
            results["loss"].append(loss.to('cpu').detach().numpy())
            results["mse"].append(mse.to('cpu').detach().numpy())
            if "pen_points" in self.dataset.keys():
                pen_points = self.local_penalization_loss_enc(batch)
                loss += self.pen_points_weight * pen_points
                recons_points = self.local_penalization_loss_dec(batch)
                loss += self.pen_points_recons * recons_points
                results["pen_points"].append(pen_points.to('cpu').detach().numpy())
                results["pen_points_recons"].append(recons_points.to('cpu').detach().numpy())
            if self.contractive_weight > 0:
                contractive = self.contractive_weight * self.contractive_loss(batch, lam=1)
                loss += contractive
                results["contractive"].append(contractive.to('cpu').detach().numpy())


        results["loss"] = np.mean(results["loss"])
        results["mse"] = np.mean(results["mse"])
        if self.contractive_weight > 0:
            results["contractive"] = np.mean(results["contractive"])
        if "pen_points" in self.dataset.keys():
            results["pen_points"] = np.mean(results["pen_points"])
            results["pen_points_recons"] = np.mean(results["pen_points_recons"])

        print("""Test loss: """, results["loss"])
        print("""Test MSE Boltzmann: """, results["mse"])

        if "pen_points" in self.dataset.keys():
            print("""Test pen points: """, results["pen_points"])
            print("""Test pen points recons: """, results["pen_points_recons"])
        return results

    def compute_cdt_avg(self, points, n_bins=40):
        if self.ae.bottleneck_dim != 1:
            raise ValueError("""Conditional averages are only implemented for AE with bottleneck of size 1""")
        X_given_z = [[[] for i in range(n_bins)] for j in range(len(self.ae.column_decoders))]
        Esp_X_given_z = [[] for i in range(len(self.ae.column_decoders))]
        self.ae.to('cpu')
        for line_dec in self.ae.line_decoders:
            line_dec.to('cpu')
        for column_dec in self.ae.column_decoders:
            column_dec.to('cpu')
        points = torch.tensor(points.astype('float32'))
        errors = [self.ae.decoded(points, i) for i in range(len(self.ae.column_decoders))]
        errors = [
            torch.sum((points.reshape(len(points), self.ae.n_lines, 1, self.ae.n_columns)
                       - error.reshape(len(points), 1, self.ae.n_lines, self.ae.n_columns)) ** 2, dim=-1)
            for error in errors
        ]
        for j, error in enumerate(errors):
            _, posi = torch.min(error, dim=-1, keepdim=True)
            mask = torch.zeros_like(error).scatter(-1, posi, 1.)
            previous = torch.zeros_like(mask[:, 0:1, :])
            for i in range(1, self.ae.n_lines):
                previous += mask[:, i - 1:i, :] * 10 ** 15
                _, posi = torch.min(previous + error, dim=-1, keepdim=True)
                mask[:, i:i + 1, :] = torch.zeros_like(error).scatter(-1, posi, 1.)[:, i:i + 1, :]
            errors[j] = (mask * error).sum(dim=[-1, -2])
        errors = torch.stack([error for error in errors])
        errors = (self.softmin_multidec(self.l_multidec * errors) * errors).sum(dim=0)

        minimum, where = torch.min(errors, dim=0)
        where = where.detach().cpu().numpy()
        xi_values = self.ae.encoded(points).detach().cpu().numpy()[:, 0]
        # equal-width bins
        z_bin = np.linspace(xi_values.min(), xi_values.max(), n_bins)
        # compute index of bin
        inds = np.digitize(xi_values, z_bin)
        # distribute train data to each bin
        for bin_idx in range(n_bins):
            for i in range(len(self.ae.column_decoders)):
                X_given_z[i][bin_idx] = points[(where == i) * (inds == bin_idx + 1), :, :]
                if len(X_given_z[i][bin_idx]) > 0:
                    Esp_X_given_z[i].append(torch.mean(X_given_z[i][bin_idx], dim=0))

        return Esp_X_given_z, X_given_z, xi_values


