"""
Simple Network Generator

A clean, straightforward approach to creating actor/critic networks
without unnecessary complexity. Just simple classes that do the job.

References:
- isaacgymenvs/learning/a2c_dict_network_builder.py: A2CBuilder
- rl_games/algos_torch/a2c_continuous.py: A2CAgent
- rl_games/algos_torch/central_value.py: CentralValueTrain
- rl_games/algos_torch/models.py: BaseModelNetwork, ModelA2CContinuousLogStd
- rl_games/algos_torch/network_builder.py: NetworkBuilder


Flow
- obs_latent = _process_observations(obs_dict)
    - Runs through MLP, CNN, PointNet, or Identity for low dim states
- either 
    - 1. out = _process_through_rnn(obs_latent)
    - 2. out = main_mlp(obs_latent)
- Actor
    - mu = mu_act(mu(out))
    - logstd = sigma_act(sigma(out))
- Critic
    - value = value_act(value(out))
- Return mu, logstd, value, states

Usage:
    actor = ActorNetwork(config)
    critic = CriticNetwork(config)
    
    # Or for backwards compatibility:
    generator = NetworkGenerator()
    actor = generator.build_actor(config)
    critic = generator.build_critic(config)
"""

import torch
import torch.nn as nn
import numpy as np

from rl.models.rnn import LSTMWithDones, GRUWithDones
import rl.models.torch_ext as torch_ext
from rl.models.spatial_softmax import SpatialSoftArgmax


def _create_initializer(func, **kwargs):
    """Create an initializer function with given parameters"""
    return lambda v: func(v, **kwargs)


def shape_whc_to_cwh(shape):
    """Convert (width, height, channels) to (channels, width, height)"""
    if len(shape) == 3:
        return (shape[2], shape[0], shape[1])
    return shape


class ActivationFactory:
    """Factory for creating activation functions"""
    
    @staticmethod
    def create(name):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'swish': nn.SiLU(),
            'gelu': nn.GELU(),
            'softplus': nn.Softplus(),
            'None': nn.Identity(),
            'none': nn.Identity(),
        }
        return activations.get(name, nn.Identity())


class InitializerFactory:
    """Factory for creating weight initializers"""
    
    @staticmethod
    def create(name='default', **kwargs):
        if name == 'const_initializer':
            return _create_initializer(nn.init.constant_, **kwargs)
        elif name == 'orthogonal_initializer':
            return _create_initializer(nn.init.orthogonal_, **kwargs)
        elif name == 'glorot_normal_initializer':
            return _create_initializer(nn.init.xavier_normal_, **kwargs)
        elif name == 'glorot_uniform_initializer':
            return _create_initializer(nn.init.xavier_uniform_, **kwargs)
        elif name == 'random_uniform_initializer':
            return _create_initializer(nn.init.uniform_, **kwargs)
        elif name == 'kaiming_normal':
            return _create_initializer(nn.init.kaiming_normal_, **kwargs)
        elif name == 'orthogonal':
            return _create_initializer(nn.init.orthogonal_, **kwargs)
        elif name == 'variance_scaling_initializer':
            return _create_initializer(torch_ext.variance_scaling_initializer, **kwargs)
        elif name == 'default':
            return lambda x: None
        else:
            return lambda x: None


class BaseNetwork(nn.Module):
    """Base network with all building functionality"""
    
    def __init__(self, config):
        super().__init__()
        
        # Factories
        self.activations_factory = ActivationFactory()
        self.init_factory = InitializerFactory()
        
        # Configuration
        self.input_shape = config['input_shape']
        self.input_params = config['input_preprocessors']
        self.mlp_config = config['mlp']
        self.value_size = config.get('value_size', 1)
        self.num_seqs = config.get('num_seqs', 1)       # NOTE: number of actors
        self.normalization = config.get('normalization', None)
        self.norm_only_first_layer = self.mlp_config.get('norm_only_first_layer', False)
        self.teacher = config.get('teacher', False)
        
        # RNN configuration
        self.has_rnn = 'rnn' in config
        if self.has_rnn:
            rnn_config = config['rnn']
            self.rnn_units = rnn_config['units']
            self.rnn_layers = rnn_config['layers']
            self.rnn_name = rnn_config['name']
            self.rnn_ln = rnn_config.get('layer_norm', False)
            self.is_rnn_before_mlp = rnn_config.get('before_mlp', False)
            self.rnn_concat_input = rnn_config.get('concat_input', False)
        
        # Build the network
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_network(self):
        """Build the entire network"""
        # Build input preprocessing networks
        self.input_networks = nn.ModuleDict()
        input_out_size = 0
        
        for input_name, input_config in self.input_params.items():
            network_layers = []
            member_input_shape = shape_whc_to_cwh(self.input_shape[input_name])
            
            # Build preprocessing layers
            if 'pnet' in input_config:
                pnet_config = input_config['pnet']
                pnet_args = {
                    'in_channels': pnet_config['in_channel'],
                    'out_channels': pnet_config['out_channel'],
                    'use_layernorm': pnet_config.get('use_layernorm', False),
                    'final_norm': pnet_config.get('final_norm', 'none'),
                    'use_projection': pnet_config.get('use_projection', True),
                }
                next_input_shape = pnet_config['out_channel']
                
            elif 'cnn' in input_config:
                cnn_config = input_config['cnn']
                cnn_args = {
                    'ctype': cnn_config['type'], 
                    'input_shape': member_input_shape, 
                    'convs': cnn_config['convs'], 
                    'activation': cnn_config['activation'], 
                    'norm_func_name': self.normalization,
                }
                cnn_layer = self._build_conv(**cnn_args)
                
                network_layers.append(cnn_layer)
                next_input_shape = self._calc_input_size(member_input_shape, cnn_layer)
                
            elif 'mlp' in input_config:
                mlp_config = input_config['mlp']
                mlp_input_size = self._calc_input_size(member_input_shape, None)
                mlp_layer = self._build_mlp(
                    mlp_input_size,
                    mlp_config['units'],
                    mlp_config['activation']
                )
                network_layers.append(mlp_layer)
                next_input_shape = mlp_config['units'][-1]
            else:
                next_input_shape = self._calc_input_size(member_input_shape, None)
            
            self.input_networks[input_name] = nn.Sequential(*network_layers)
            input_out_size += next_input_shape
        
        # Build main MLP
        in_mlp_shape = input_out_size
        out_size = self.mlp_config['units'][-1] if self.mlp_config['units'] else in_mlp_shape
        
        # Handle RNN
        if self.has_rnn:
            if not self.is_rnn_before_mlp:
                rnn_in_size = out_size
                out_size = self.rnn_units
                if self.rnn_concat_input:
                    rnn_in_size += in_mlp_shape
            else:
                rnn_in_size = in_mlp_shape
                in_mlp_shape = self.rnn_units
            
            self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
            if self.rnn_ln:
                self.layer_norm = nn.LayerNorm(self.rnn_units)
        
        self.main_mlp = self._build_mlp(in_mlp_shape, self.mlp_config['units'], self.mlp_config['activation'])
        self.output_size = out_size
    
    def _calc_input_size(self, input_shape, cnn_layers=None):
        """Calculate flattened input size"""
        if cnn_layers is None:
            if len(input_shape) == 1:
                return input_shape[0]
            else:
                return int(np.prod(input_shape))
        else:
            with torch.no_grad():
                dummy_input = torch.randn(1, *input_shape)
                dummy_output = cnn_layers(dummy_input)
                return dummy_output.flatten(1).size(1)
    
    def _build_mlp(self, input_size, units, activation):
        """Build MLP layers"""
        if not units:
            return nn.Identity()
        
        layers = []
        in_size = input_size
        
        for unit in units:
            layers.append(nn.Linear(in_size, unit))
            layers.append(self.activations_factory.create(activation))
            in_size = unit
        
        return nn.Sequential(*layers)
    
    def _build_conv(self, ctype, **kwargs):
        print('conv_name:', ctype)

        if ctype == 'conv2d':
            return self._build_cnn2d(**kwargs)
        if ctype == 'conv2d_spatial_softargmax':
            return self._build_cnn2d(add_spatial_softmax=True, **kwargs)
        if ctype == 'conv2d_flatten':
            return self._build_cnn2d(add_flatten=True, **kwargs)
        if ctype == 'coord_conv2d':
            return self._build_cnn2d(conv_func=torch_ext.CoordConv2d, **kwargs)
        if ctype == 'conv1d':
            return self._build_cnn1d(**kwargs)

    def _build_cnn2d(self, input_shape, convs, activation, conv_func=torch.nn.Conv2d, norm_func_name=None, add_spatial_softmax=False, add_flatten=False):
        in_channels = input_shape[0]
        layers = []
        for conv in convs:
            layers.append(conv_func(in_channels=in_channels, 
            out_channels=conv['filters'], 
            kernel_size=conv['kernel_size'], 
            stride=conv['strides'], padding=conv['padding']))
            conv_func=torch.nn.Conv2d
            act = self.activations_factory.create(activation)
            layers.append(act)
            in_channels = conv['filters']
            if norm_func_name == 'layer_norm':
                layers.append(torch_ext.LayerNorm2d(in_channels))
            elif norm_func_name == 'batch_norm':
                layers.append(torch.nn.BatchNorm2d(in_channels))
        if add_spatial_softmax:
            layers.append(SpatialSoftArgmax(normalize=True))
        if add_flatten:
            layers.append(torch.nn.Flatten())
        return nn.Sequential(*layers)
    
    def _build_cnn1d(self, input_shape, convs, activation):
        """Build 1D CNN layers"""
        in_channels = input_shape[0]
        layers = []
        
        for conv in convs:
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv['filters'],
                kernel_size=conv['kernel_size'],
                stride=conv['strides'],
                padding=conv['padding']
            ))
            layers.append(self.activations_factory.create(activation))
            in_channels = conv['filters']
        
        return nn.Sequential(*layers)
    


    def _build_rnn(self, name, input_size, units, layers):
        """Build RNN layer"""
        if name == 'lstm':
            return LSTMWithDones(input_size, units, layers)
        elif name == 'gru':
            return GRUWithDones(input_size, units, layers)
        elif name == 'identity':
            return torch_ext.IdentityRNN(input_size, units)
        else:
            raise ValueError(f"Unsupported RNN type: {name}")
    
    def _initialize_weights(self):
        """Initialize network weights"""
        mlp_init = self.init_factory.create(**self.mlp_config['initializer'])
        
        # Initialize CNN weights if present
        cnn_init = None
        for input_name, input_config in self.input_params.items():
            if 'cnn' in input_config:
                cnn_init = self.init_factory.create(**input_config['cnn']['initializer'])
                break
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)) and cnn_init is not None:
                cnn_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if mlp_init is not None:
                    mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
    
    def _process_student_observations(self, obs_dict):
        """Process dictionary observations through input networks"""
        obs = obs_dict['obs']
        obs = {**obs}  # Copy to avoid modifying original
        
        # Handle CNN input format conversion
        for input_name, input_config in self.input_params.items():
            if 'cnn' in input_config and len(obs[input_name].shape) == 4:
                # Convert (B, W, H, C) to (B, C, W, H)
                obs[input_name] = obs[input_name].permute((0, 3, 1, 2))
        
        # Process each input through its preprocessing network
        input_out_vals = []
        for input_name in self.input_params.keys():
            out_dim = self.input_networks[input_name](obs[input_name])
            input_out_vals.append(out_dim.contiguous().view(out_dim.size(0), -1))
        
        return torch.cat(input_out_vals, dim=-1)

    def _process_teacher_observations(self, obs_dict):
        states = obs_dict['states']
        states = {**states}

        input_out_vals = []
        for input_name in self.input_params.keys():
            out_dim = self.input_networks[input_name](states[input_name])
            input_out_vals.append(out_dim.contiguous().view(out_dim.size(0), -1))

        return torch.cat(input_out_vals, dim=-1)
    
    def _process_through_rnn(self, out, obs_dict):
        """Process features through RNN if present"""
        if not self.has_rnn:
            return out, None
        
        states = obs_dict.get('rnn_states', None)   # [(num_layers, num_actors, hidden_size), ...]
        seq_length = obs_dict.get('seq_length', 1)  # NOTE: refers to the length of the obs sequence
        dones = obs_dict.get('dones', None)
        bptt_len = obs_dict.get('bptt_len', 0)
        
        out_in = out
        if not self.is_rnn_before_mlp:
            out = self.main_mlp(out)
            if self.rnn_concat_input:
                out = torch.cat([out, out_in], dim=1)

        batch_size = out.size()[0]
        num_seqs = batch_size // seq_length
        out = out.reshape(num_seqs, seq_length, -1)

        # Adjust states for current batch size
        if states is None:
            if self.rnn_name == 'lstm':
                h = torch.zeros((self.rnn_layers, num_seqs, self.rnn_units), device=out.device)
                c = torch.zeros((self.rnn_layers, num_seqs, self.rnn_units), device=out.device)
                states = (h, c)
            else:
                states = torch.zeros((self.rnn_layers, num_seqs, self.rnn_units), device=out.device)
        elif isinstance(states, (list, tuple)) and len(states) == 1:
            states = states[0]

        out = out.transpose(0, 1)
        # NOTE: this will zero out the hidden states of the RnnWithDones
        if dones is not None:
            dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)
        
        out, states = self.rnn(out, states, dones, bptt_len)
        out = out.transpose(0, 1).contiguous().reshape(out.size()[0] * out.size()[1], -1)

        if self.rnn_ln:
            out = self.layer_norm(out)
        
        if self.is_rnn_before_mlp:
            out = self.main_mlp(out)
        
        if not isinstance(states, tuple):
            states = (states,)
        
        return out, states
    
    def is_rnn(self):
        return self.has_rnn

    def get_default_rnn_state(self):
        if not self.has_rnn:
            return None

        if self.rnn_name == 'identity':
            rnn_units = 1
        else:
            rnn_units = self.rnn_units
        if self.rnn_name == 'lstm':
            return (torch.zeros((self.rnn_layers, self.num_seqs, rnn_units)), 
                    torch.zeros((self.rnn_layers, self.num_seqs, rnn_units)))
        else:
            return (torch.zeros((self.rnn_layers, self.num_seqs, rnn_units)),)             

class ActorNetwork(BaseNetwork):
    """Actor network that outputs action mean and log standard deviation"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Action space configuration
        self.actions_num = config['actions_num']
        space_config = config['space']['continuous']
        
        # Action output layers
        self.mu = nn.Linear(self.output_size, self.actions_num)
        self.mu_act = self.activations_factory.create(space_config['mu_activation'])
        
        self.fixed_sigma = space_config['fixed_sigma']
        self.sigma_act = self.activations_factory.create(space_config['sigma_activation'])
        
        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32))
        else:
            self.sigma = nn.Linear(self.output_size, self.actions_num)
        
        # Initialize action layers
        mu_init = self.init_factory.create(**space_config['mu_init'])
        sigma_init = self.init_factory.create(**space_config['sigma_init'])
        
        if mu_init is not None:
            mu_init(self.mu.weight)
        if self.fixed_sigma:
            if sigma_init is not None:
                sigma_init(self.sigma)
        else:
            if sigma_init is not None:
                sigma_init(self.sigma.weight)
    
    def forward(self, obs_dict):
        """Forward pass for actor network"""
        # Process observations
        if self.teacher:
            out = self._process_teacher_observations(obs_dict)
        else:
            out = self._process_student_observations(obs_dict)
        
        # Process through RNN if present
        if self.has_rnn:
            out, states = self._process_through_rnn(out, obs_dict)
        else:
            out = self.main_mlp(out)
            states = None
        
        # Generate action outputs
        mu = self.mu_act(self.mu(out))
        
        if self.fixed_sigma:
            logstd = self.sigma
        else:
            logstd = self.sigma_act(self.sigma(out))
        
        return mu, logstd, states

    def forward_with_features(self, obs_dict):
        """Forward pass for actor network"""
        # Process observations
        if self.teacher:
            out = self._process_teacher_observations(obs_dict)
        else:
            out = self._process_student_observations(obs_dict)
        
        # Process through RNN if present
        if self.has_rnn:
            features, states = self._process_through_rnn(out, obs_dict)
        else:
            features = self.main_mlp(out)
            states = None
        
        # Generate action outputs
        mu = self.mu_act(self.mu(features))
        
        if self.fixed_sigma:
            logstd = self.sigma
        else:
            logstd = self.sigma_act(self.sigma(features))
        
        return mu, logstd, states, features


class CriticNetwork(BaseNetwork):
    """Critic network that outputs value estimates"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Value output layer
        value_activation = config.get('value_activation', 'None')
        self.value = nn.Linear(self.output_size, self.value_size)
        self.value_act = self.activations_factory.create(value_activation)
    
    def forward(self, obs_dict):
        """Forward pass for critic network"""
        # Process observations
        if self.teacher:
            out = self._process_teacher_observations(obs_dict)
        else:
            out = self._process_student_observations(obs_dict)
        
        # Process through RNN if present
        if self.has_rnn:
            out, states = self._process_through_rnn(out, obs_dict)
        else:
            out = self.main_mlp(out)
            states = None
        
        # Generate value output
        value = self.value_act(self.value(out))
        
        return value, states

class QNetwork(BaseNetwork):
    """A single Q network that outputs state-action (Q) values for SAC"""

    def __init__(self, config):
        super().__init__(config)
        
        # For SAC, Q-network outputs a single Q-value given (state, action) pair
        # The action is concatenated with state features before the final Q layer
        self.actions_num = config.get('actions_num', 0)
        q_activation = config.get('q_activation', 'None')
        
        # Q output layer takes state features + action as input
        self.q = nn.Linear(self.output_size + self.actions_num, 1)
        self.q_act = self.activations_factory.create(q_activation)
    
    def forward(self, obs_dict, action=None):
        """
        Forward pass for Q-network.
        
        Args:
            obs_dict: Dictionary containing observations/states
            action: Action tensor [B, action_dim] to concatenate with state features
        
        Returns:
            q: Q-value estimate [B, 1]
            states: RNN states (if using RNN, else None)
        """
        obs_dict['states']['action'] = action
        obs_dict['obs']['action'] = action

        if self.teacher:
            out = self._process_teacher_observations(obs_dict)
        else:
            out = self._process_student_observations(obs_dict)

        if self.has_rnn:
            out, states = self._process_through_rnn(out, obs_dict)
        else:
            out = self.main_mlp(out)
            states = None
        
        q = self.q_act(self.q(out))

        return q, states


# Utility functions
def build_actor(config):
    """Build an actor network from config"""
    return ActorNetwork(config)


def build_critic(config):
    """Build a critic network from config"""
    return CriticNetwork(config)


class NetworkGenerator:
    """Simple network generator for backwards compatibility"""
    
    def build_actor(self, config):
        return build_actor(config)
    
    def build_critic(self, config):
        return build_critic(config)


def main():
    """Test the simple network generator"""
    
    # Test configuration
    config = {
        'input_shape': {
            'camera': (84, 84, 3),
            'tactile': (32, 10),
            'joint_pos': (7,)
        },
        'input_preprocessors': {
            'camera': {
                'cnn': {
                    'type': 'conv2d',
                    'convs': [
                        {'filters': 32, 'kernel_size': 8, 'strides': 4, 'padding': 0},
                        {'filters': 64, 'kernel_size': 4, 'strides': 2, 'padding': 0},
                        {'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 0}
                    ],
                    'activation': 'relu',
                    'initializer': {'name': 'default'}
                }
            },
            'tactile': {
                'cnn': {
                    'type': 'conv1d',
                    'convs': [
                        {'filters': 16, 'kernel_size': 3, 'strides': 1, 'padding': 1},
                        {'filters': 32, 'kernel_size': 3, 'strides': 1, 'padding': 1}
                    ],
                    'activation': 'relu',
                    'initializer': {'name': 'default'}
                }
            },
            'joint_pos': {
                'mlp': {
                    'units': [64, 32],
                    'activation': 'relu'
                }
            }
        },
        'mlp': {
            'units': [256, 128, 64],
            'activation': 'relu',
            'initializer': {'name': 'default'},
            'norm_only_first_layer': False
        },
        'actions_num': 6,
        'space': {
            'continuous': {
                'mu_activation': 'None',
                'sigma_activation': 'None',
                'mu_init': {'name': 'default'},
                'sigma_init': {'name': 'default'},
                'fixed_sigma': True
            }
        },
        'value_size': 1,
        'value_activation': 'None',
        'num_seqs': 1,
        'normalization': None
    }
    
    print("Creating simple networks...")
    
    # Test direct instantiation
    actor = ActorNetwork(config)
    critic = CriticNetwork(config)
    
    print(f"Actor: {type(actor).__name__}")
    print(f"Critic: {type(critic).__name__}")
    print(f"Actor is RNN: {actor.is_rnn()}")
    print(f"Critic is RNN: {critic.is_rnn()}")
    
    # Test with sample data
    batch_size = 4
    obs_dict = {
        'obs': {
            'camera': torch.randn(batch_size, 84, 84, 3),
            'tactile': torch.randn(batch_size, 32, 10),
            'joint_pos': torch.randn(batch_size, 7)
        }
    }
    
    print(f"\nTesting forward pass with batch_size={batch_size}...")
    
    with torch.no_grad():
        # Test actor
        mu, logstd, actor_states = actor(obs_dict)
        print(f"Actor outputs:")
        print(f"  Action mean (mu): {mu.shape}")
        print(f"  Action logstd: {logstd.shape}")
        
        # Test critic
        value, critic_states = critic(obs_dict)
        print(f"Critic outputs:")
        print(f"  Value: {value.shape}")
    
    print("\n✓ Simple network generator test completed successfully!")
    print("✓ Clean, straightforward implementation")
    print("✓ No unnecessary nested classes")
    print("✓ Direct instantiation: ActorNetwork(config), CriticNetwork(config)")


if __name__ == "__main__":
    main()
