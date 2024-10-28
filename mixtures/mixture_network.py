from typing import Callable, List

import torch

from inr.network import BaseNetwork, Sequential


class MixtureNetwork(BaseNetwork):
    """
    A simple M-to-N Mixture Network.

    """
    def __init__(
            self,
            input_features: int,
            output_features: int,
            hidden_features: List[int],
            n_weights: int,
            n_images: int,
            compute_function: Callable = None,
            **kwargs
        ):
        super().__init__(**kwargs)

        self.input_features = input_features
        self.output_features = output_features
        self.hidden_features = hidden_features
        self.n_weights = n_weights
        self.n_images = n_images

        # Generate `n_weights` sets of parameters
        self.weights_list = torch.nn.ParameterList()
        self._get_n_weights(**kwargs)
        self.trainable_parameters = self.parameters()

        # Compute `n_images` sets of parameters
        self.params_list = self._get_n_params(**kwargs)

        self.compute_function = compute_function

        # Create the network
        self.network = Sequential(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_features=self.hidden_features,
            **kwargs
        )

        # Remove gradients from the network
        for param in self.network.parameters():
            param.requires_grad = False

    def _get_n_weights(self, **kwargs):
        for i in range(self.n_weights):
            net = Sequential(
                input_features=self.input_features,
                output_features=self.output_features,
                hidden_features=self.hidden_features,
                **kwargs
            )

            weights = torch.nn.ParameterDict()
            for key, value in net.state_dict().items():
                param_name = f"weightlist.{i}.{key}"
                param = torch.nn.Parameter(value)
                weights[param_name.replace('.', '_')] = param
            self.weights_list.append(weights)

    def _get_n_params(self, **kwargs):
        params_list = []
        for _ in range(self.n_images):
            ref_params = self.weights_list[0]
            params = {}
            for key, value in ref_params.items():
                stripped_key = key.replace('_', '.').split('.')[2:]
                stripped_key = '.'.join(stripped_key)
                params[stripped_key] = torch.zeros_like(value)
            params_list.append(params)

        return params_list

    def to(self, device):
        self.device = device
        self.network.to(device)
        for i in range(self.n_weights):
            for key in self.weights_list[i].keys():
                self.weights_list[i][key] = self.weights_list[i][key].to(device)
        for i in range(self.n_images):
            for key in self.params_list[i].keys():
                self.params_list[i][key] = self.params_list[i][key].to(device)

        return self

    def train(self, mode=True):
        self.network.train(mode)
        return self

    def eval(self):
        self.network.eval()
        return self

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        self._compute_current_params()
        return_list = []
        for params in self.params_list:
            return_list.append(self.network(x, param_dict=params))

        return torch.stack(return_list, dim=1)

    def _compute_current_params(self):
        """
        Compute the current set of parameters.
        """
        param_keys = list(self.params_list[0].keys())
        for key in param_keys:
            computed_params = self.compute_function([self.weights_list[i][f"weightlist_{i}_{key.replace('.', '_')}"] for i in range(self.n_weights)])
            for i in range(self.n_images):
                self.params_list[i][key] = computed_params[i]
