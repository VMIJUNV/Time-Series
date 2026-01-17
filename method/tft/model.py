import sys
sys.path.append(".")

from models import GatedLinearUnit
from models import GateAddNormNetwork
from models import GatedResidualNetwork 
from models import ScaledDotProductAttention
from models import InterpretableMultiHeadAttention
from models import VariableSelectionNetwork
import torch
from torch import nn

from quantile_loss import QuantileLossCalculator
from quantile_loss import NormalizedQuantileLossCalculator

from dataclasses import dataclass, field

from utils.more_utils import BaseDataclass

@dataclass
class Hparams(BaseDataclass):
    input_size: int = None
    output_size: int = None
    num_encoder_steps: int = None
    quantiles: list = field(default_factory=lambda: [0.5])

    category_counts: list = field(default_factory=lambda: [])
    input_obs_loc: list = field(default_factory=lambda: [])
    static_input_loc: list = field(default_factory=lambda: [])
    known_regular_inputs: list = field(default_factory=lambda: [])
    known_categorical_inputs: list = field(default_factory=lambda: [])

    hidden_layer_size: int = 160
    dropout_rate: float = 0.1
    num_heads: int = 4
    num_lstm_layers: int = 1


class TemporalFusionTransformer(nn.Module):
    def __init__(self, hparams: Hparams):
        super(TemporalFusionTransformer, self).__init__()
        
        self.hparams = hparams
        
        # Data parameters
        self.input_size = hparams.input_size
        self.output_size = hparams.output_size
        self.category_counts = hparams.category_counts
        self.num_categorical_variables = len(self.category_counts)
        self.num_regular_variables = self.input_size - self.num_categorical_variables

        # Relevant indices for TFT
        self._input_obs_loc = hparams.input_obs_loc
        self._static_input_loc = hparams.static_input_loc
        self._known_regular_input_idx = hparams.known_regular_inputs
        self._known_categorical_input_idx = hparams.known_categorical_inputs
        
        self.num_non_static_historical_inputs = self.get_historical_num_inputs()
        self.num_non_static_future_inputs = self.get_future_num_inputs()
        
        # Network params
        self.quantiles = hparams.quantiles
        self.num_lstm_layers = hparams.num_lstm_layers
        self.hidden_layer_size = hparams.hidden_layer_size
        self.dropout_rate = hparams.dropout_rate
        self.num_encoder_steps = hparams.num_encoder_steps
        self.num_heads = hparams.num_heads

        # Serialisation options
#         self._temp_folder = os.path.join(params['model_folder'], 'tmp')
#         self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None
            
        self.train_criterion = QuantileLossCalculator(self.quantiles, self.output_size)
        self.test_criterion = NormalizedQuantileLossCalculator(self.quantiles, self.output_size)

        # Build model
        ## Build embeddings
        self.build_embeddings()
        
        ## Build Static Contex Networks
        self.build_static_context_networks()
        
        ## Building Variable Selection Networks
        self.build_variable_selection_networks()
        
        ## Build Lstm
        self.build_lstm()
        
        ## Build GLU for after lstm encoder decoder and layernorm
        self.build_post_lstm_gate_add_norm()
        
        ## Build Static Enrichment Layer
        self.build_static_enrichment()
        
        ## Building decoder multihead attention
        self.build_temporal_self_attention()
        
        ## Building positionwise decoder
        self.build_position_wise_feed_forward()
        
        ## Build output feed forward
        self.build_output_feed_forward()
        
        ## Initializing remaining weights
        self.init_weights()
        
    def init_weights(self):
        for name, p in self.named_parameters():
            if ('lstm' in name and 'ih' in name) and 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
#                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif ('lstm' in name and 'hh' in name) and 'bias' not in name:
        
                torch.nn.init.orthogonal_(p)
            
            elif 'lstm' in name and 'bias' in name:
                torch.nn.init.zeros_(p)
        
    def get_historical_num_inputs(self):
        
        obs_inputs = [i for i in self._input_obs_loc]
        
        known_regular_inputs = [i for i in self._known_regular_input_idx
                                    if i not in self._static_input_loc]
            
        known_categorical_inputs = [i for i in self._known_categorical_input_idx
                                    if i + self.num_regular_variables not in self._static_input_loc]
        
        wired_embeddings = [i for i in range(self.num_categorical_variables)
                                if i not in self._known_categorical_input_idx 
                                and i not in self._input_obs_loc]

        unknown_inputs = [i for i in range(self.num_regular_variables)
                            if i not in self._known_regular_input_idx
                            and i not in self._input_obs_loc]

        return len(obs_inputs+known_regular_inputs+known_categorical_inputs+wired_embeddings+unknown_inputs)
    
    def get_future_num_inputs(self):
            
        known_regular_inputs = [i for i in self._known_regular_input_idx
                                if i not in self._static_input_loc]
            
        known_categorical_inputs = [i for i in self._known_categorical_input_idx
                                    if i + self.num_regular_variables not in self._static_input_loc]

        return len(known_regular_inputs + known_categorical_inputs)
    
    def build_embeddings(self):
        self.categorical_var_embeddings = nn.ModuleList([nn.Embedding(self.category_counts[i], self.hidden_layer_size) for i in range(self.num_categorical_variables)])

        self.regular_var_embeddings = nn.ModuleList([nn.Linear(1,self.hidden_layer_size) for i in range(self.num_regular_variables)])

    def build_variable_selection_networks(self):
        
        self.static_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,
                                                   input_size = self.hidden_layer_size * len(self._static_input_loc),
                                                    output_size = len(self._static_input_loc),
                                                    dropout_rate = self.dropout_rate)
        
        self.temporal_historical_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,
                                                                input_size = self.hidden_layer_size *self.num_non_static_historical_inputs,
                                                                output_size = self.num_non_static_historical_inputs,
                                                                dropout_rate = self.dropout_rate,
                                                                additional_context=self.hidden_layer_size)
        
        self.temporal_future_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,
                                                            input_size = self.hidden_layer_size *self.num_non_static_future_inputs,
                                                            output_size = self.num_non_static_future_inputs,
                                                            dropout_rate = self.dropout_rate,
                                                            additional_context=self.hidden_layer_size)
        
    def build_static_context_networks(self):
        
        self.static_context_variable_selection_grn = GatedResidualNetwork(self.hidden_layer_size,dropout_rate=self.dropout_rate)
        
        self.static_context_enrichment_grn = GatedResidualNetwork(self.hidden_layer_size,dropout_rate=self.dropout_rate)

        self.static_context_state_h_grn = GatedResidualNetwork(self.hidden_layer_size,dropout_rate=self.dropout_rate)
        
        self.static_context_state_c_grn = GatedResidualNetwork(self.hidden_layer_size,dropout_rate=self.dropout_rate)
        
    def build_lstm(self):
        self.historical_lstm = nn.LSTM(input_size = self.hidden_layer_size,
                                        hidden_size = self.hidden_layer_size,
                                        num_layers=self.num_lstm_layers,
                                        batch_first = True)
        self.future_lstm = nn.LSTM(input_size = self.hidden_layer_size,
                                    hidden_size = self.hidden_layer_size,
                                    num_layers=self.num_lstm_layers,
                                    batch_first = True)
        
    def build_post_lstm_gate_add_norm(self):
        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                                self.hidden_layer_size,
                                                                self.dropout_rate,
                                                                activation = None)
        
    def build_static_enrichment(self):
        self.static_enrichment = GatedResidualNetwork(self.hidden_layer_size,
                                                    dropout_rate = self.dropout_rate,
                                                    additional_context=self.hidden_layer_size)
        
    def build_temporal_self_attention(self):
        self.self_attn_layer = InterpretableMultiHeadAttention(n_head = self.num_heads, 
                                                                d_model = self.hidden_layer_size,
                                                                dropout = self.dropout_rate)
        
        self.post_attn_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                            self.hidden_layer_size,
                                                            self.dropout_rate,
                                                            activation = None)
        
    def build_position_wise_feed_forward(self):
        self.GRN_positionwise = GatedResidualNetwork(self.hidden_layer_size,
                                                        dropout_rate = self.dropout_rate)
        
        self.post_tfd_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                            self.hidden_layer_size,
                                                            self.dropout_rate,
                                                            activation = None)
        
    def build_output_feed_forward(self):
        self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size,self.output_size * len(self.quantiles))

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.
        Args:
        self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1]
        bs = self_attn_inputs.shape[0]
        mask = torch.cumsum(torch.eye(len_s,device=self_attn_inputs.device), 0)
        mask = mask.repeat(bs,1,1).to(torch.float32)
        return mask
    
    def get_tft_embeddings(self, regular_inputs, categorical_inputs):
        # Static input
        if self._static_input_loc:
            static_regular_inputs = [self.regular_var_embeddings[i](regular_inputs[:, 0, i:i + 1]) 
                                    for i in range(self.num_regular_variables)
                                    if i in self._static_input_loc]
            
            static_categorical_inputs = [self.categorical_var_embeddings[i](categorical_inputs[..., i])[:,0,:] 
                                        for i in range(self.num_categorical_variables)
                                        if i + self.num_regular_variables in self._static_input_loc]
            
            static_inputs = torch.stack(static_regular_inputs + static_categorical_inputs, axis = 1)
        else:
            static_inputs = None
            
        # Target input
        obs_inputs = torch.stack([self.regular_var_embeddings[i](regular_inputs[..., i:i + 1])
                                    for i in self._input_obs_loc], axis=-1)
        
        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(self.num_categorical_variables):
            if i not in self._known_categorical_input_idx \
            and i not in self._input_obs_loc:
                e = self.categorical_var_embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(self.num_regular_variables):
            if i not in self._known_regular_input_idx \
            and i not in self._input_obs_loc:
                e = self.regular_var_embeddings[i](regular_inputs[..., i:i + 1])
                unknown_inputs.append(e)
                
        if unknown_inputs + wired_embeddings:
            unknown_inputs = torch.stack(unknown_inputs + wired_embeddings, axis=-1)
        else:
            unknown_inputs = None
            
        # A priori known inputs
        known_regular_inputs = [self.regular_var_embeddings[i](regular_inputs[..., i:i + 1])
                                for i in self._known_regular_input_idx
                                if i not in self._static_input_loc]
        
        known_categorical_inputs = [self.categorical_var_embeddings[i](categorical_inputs[..., i])
                                    for i in self._known_categorical_input_idx
                                    if i + self.num_regular_variables not in self._static_input_loc]

        known_combined_layer = torch.stack(known_regular_inputs + known_categorical_inputs, axis=-1)
        
        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs
        
    def forward(self, all_inputs):

        regular_inputs = all_inputs[:, :, :self.num_regular_variables].to(torch.float)
        categorical_inputs = all_inputs[:, :, self.num_regular_variables:].to(torch.long)
        
        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_tft_embeddings(regular_inputs, categorical_inputs)
        
        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = torch.cat([
                unknown_inputs[:, :self.num_encoder_steps, :],
                known_combined_layer[:, :self.num_encoder_steps, :],
                obs_inputs[:, :self.num_encoder_steps, :]
            ], axis=-1)
        else:
            historical_inputs = torch.cat([
                known_combined_layer[:, :self.num_encoder_steps, :],
                obs_inputs[:, :self.num_encoder_steps, :]
            ], axis=-1)
        
        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, self.num_encoder_steps:, :]

        static_encoder, sparse_weights = self.static_vsn(static_inputs)
        static_context_variable_selection = self.static_context_variable_selection_grn(static_encoder)
        static_context_enrichment = self.static_context_enrichment_grn(static_encoder)
        static_context_state_h = self.static_context_state_h_grn(static_encoder)
        static_context_state_c = self.static_context_state_c_grn(static_encoder)
        
        historical_features, historical_flags = self.temporal_historical_vsn((historical_inputs,static_context_variable_selection))
        
        future_features, future_flags = self.temporal_future_vsn((future_inputs,static_context_variable_selection))
        
        static_context_state_h_= static_context_state_h.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
        static_context_state_c_ = static_context_state_c.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
        
        history_lstm, (state_h, state_c) = self.historical_lstm(historical_features,(static_context_state_h_,static_context_state_c_))
        
        future_lstm, _ = self.future_lstm(future_features,(state_h,state_c))
        
        # Apply gated skip connection
        input_embeddings = torch.cat((historical_features, future_features), axis=1)
        
        lstm_layer = torch.cat((history_lstm, future_lstm), axis=1)
        
        temporal_feature_layer = self.post_seq_encoder_gate_add_norm(lstm_layer, input_embeddings)
        
        # Static enrichment layers
        expanded_static_context = static_context_enrichment.unsqueeze(1)
        
        enriched = self.static_enrichment((temporal_feature_layer, expanded_static_context))
        x, self_att = self.self_attn_layer(enriched, enriched, enriched,mask = self.get_decoder_mask(enriched))
        
        x = self.post_attn_gate_add_norm(x, enriched)
        
        # Nonlinear processing on outputs
        decoder = self.GRN_positionwise(x)

        # Final skip connection
        transformer_layer = self.post_tfd_gate_add_norm(decoder, temporal_feature_layer)
        outputs = self.output_feed_forward(transformer_layer[..., self.num_encoder_steps:, :])
        
        return outputs
