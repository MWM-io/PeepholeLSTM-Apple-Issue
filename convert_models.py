import torch as th
import coremltools as ct
from typing import Tuple
from tensorflow.python.keras.layers.recurrent import PeepholeLSTMCell
from keras.layers import RNN, Dense, Bidirectional
from keras.models import Sequential
from torch.utils import mobile_optimizer


# =====================#
#      TORCHSCRIPT     #
# =====================#

class LSTMStep(th.nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Input gate
        self.input_weights = th.nn.Parameter(th.empty(hidden_size, input_size))
        self.input_recurrent_weights = th.nn.Parameter(
            th.empty(hidden_size, hidden_size))
        self.input_peephole_weights = th.nn.Parameter(th.empty(hidden_size))
        self.input_bias = th.nn.Parameter(th.empty(hidden_size))

        # Forget gate
        self.forget_weights = th.nn.Parameter(
            th.empty(hidden_size, input_size))
        self.forget_recurrent_weights = th.nn.Parameter(
            th.empty(hidden_size, hidden_size))
        self.forget_peephole_weights = th.nn.Parameter(th.empty(hidden_size))
        self.forget_bias = th.nn.Parameter(th.empty(hidden_size))

        # Cell gate
        self.cell_weights = th.nn.Parameter(th.empty(hidden_size, input_size))
        self.cell_recurrent_weights = th.nn.Parameter(
            th.empty(hidden_size, hidden_size))
        self.cell_bias = th.nn.Parameter(th.empty(hidden_size))

        # Output gate
        self.output_weights = th.nn.Parameter(
            th.empty(hidden_size, input_size))
        self.output_recurrent_weights = th.nn.Parameter(
            th.empty(hidden_size, hidden_size))
        self.output_peephole_weights = th.nn.Parameter(th.empty(hidden_size))
        self.output_bias = th.nn.Parameter(th.empty(hidden_size))

        self.sigmoid = th.nn.Sigmoid()
        self.tanh = th.nn.Tanh()

    def forward(self,
                xt: th.Tensor,
                ht: th.Tensor,
                ct: th.Tensor
                ) -> Tuple[th.Tensor, th.Tensor]:
        it = self.sigmoid(th.matmul(self.input_weights, xt) +
                          th.matmul(self.input_recurrent_weights, ht) +
                          self.input_peephole_weights * ct +
                          self.input_bias)
        ft = self.sigmoid(th.matmul(self.forget_weights, xt) +
                          th.matmul(self.forget_recurrent_weights, ht) +
                          self.forget_peephole_weights * ct +
                          self.forget_bias)
        gt = self.tanh(th.matmul(self.cell_weights, xt) +
                       th.matmul(self.cell_recurrent_weights, ht) +
                       self.cell_bias)
        ct = ft * ct + it * gt

        ot = self.sigmoid(th.matmul(self.output_weights, xt) +
                          th.matmul(self.output_recurrent_weights, ht) +
                          self.output_peephole_weights * ct +
                          self.output_bias)
        ht = ot * self.tanh(ct)
        return ht, ct


class LSTMCell(th.nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        x0 = th.zeros(input_size)
        h0 = th.zeros(hidden_size)
        c0 = th.zeros(hidden_size)
        self.lstm_step = th.jit.trace(
            LSTMStep(input_size, hidden_size), (x0, h0, c0))

    def forward(self,
                input: th.Tensor
                ) -> th.Tensor:
        ht, ct = th.zeros(self.hidden_size), th.zeros(self.hidden_size)
        out = []
        for xt in input:
            ht, ct = self.lstm_step(xt, ht, ct)
            out.append(ht.unsqueeze(0))
        return th.cat(out, dim=0)


class ParallelBiLSTMCell(th.nn.Module):

    def __init__(self, input_size, hidden_size, parallel=False):
        super().__init__()

        self.fwd = LSTMCell(input_size, hidden_size)
        self.bwd = LSTMCell(input_size, hidden_size)

    def forward(self,
                input: th.Tensor
                ) -> th.Tensor:
        fwd_out = th.jit.fork(self.fwd, input)
        bwd_out = th.flip(self.bwd(th.flip(input, [0])), [0])
        fwd_out = th.jit.wait(fwd_out)
        return th.cat([fwd_out, bwd_out], dim=1)


class BiLSTMCell(th.nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fwd = LSTMCell(input_size, hidden_size)
        self.bwd = LSTMCell(input_size, hidden_size)

    def forward(self,
                input: th.Tensor
                ) -> th.Tensor:
        fwd_out = self.fwd(input)
        bwd_out = th.flip(self.bwd(th.flip(input, [0])), [0])
        return th.cat([fwd_out, bwd_out], dim=1)


class FeedForwardLayer(th.nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.weights = th.nn.Parameter(th.empty(input_size, 1))
        self.bias = th.nn.Parameter(th.empty(1))
        self.sigmoid = th.nn.Sigmoid()

    def forward(self,
                input: th.Tensor
                ) -> th.Tensor:
        return self.sigmoid(th.matmul(input, self.weights) + self.bias)


class RNNTorchScript(th.nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=False, parallel=False):
        super().__init__()

        if bidirectional:
            if parallel:
                cell = ParallelBiLSTMCell
            else:
                cell = BiLSTMCell
            self.lstm0 = cell(input_size, hidden_size)
            self.lstm1 = cell(2*hidden_size, hidden_size)
            self.lstm2 = cell(2*hidden_size, hidden_size)
            self.ffl = FeedForwardLayer(2*hidden_size)
        else:
            self.lstm0 = LSTMCell(input_size, hidden_size)
            self.lstm1 = LSTMCell(hidden_size, hidden_size)
            self.lstm2 = LSTMCell(hidden_size, hidden_size)
            self.ffl = FeedForwardLayer(hidden_size)

    def forward(self,
                input: th.Tensor
                ) -> th.Tensor:
        x = self.lstm0(input)
        x = self.lstm1(x)
        x = self.lstm2(x)
        return self.ffl(x)

# Convert parallel bidirectional model to TorchScript
birnn = RNNTorchScript(314, 25, bidirectional=True, parallel=True)
birnn_ts = th.jit.script(birnn)
birnn_ts_opti = mobile_optimizer.optimize_for_mobile(birnn_ts)
birnn_ts_opti.save(
    'TorchScriptVSCoreML/TorchScriptVSCoreML/MLModel/parallel_birnn.pt')


# ================ #
#      COREML      #
# ================ #

# Convert bidirectional to CoreML
length = None
n_features = 314
n_hidden = 25

birnn_keras = Sequential()
birnn_keras.add(Bidirectional(RNN(PeepholeLSTMCell(
    units=n_hidden), return_sequences=True), input_shape=(length, n_features)))
birnn_keras.add(Bidirectional(RNN(PeepholeLSTMCell(units=n_hidden), return_sequences=True)))
birnn_keras.add(Bidirectional(RNN(PeepholeLSTMCell(units=n_hidden), return_sequences=True)))
birnn_keras.add(Dense(1, activation='softmax'))
birnn_keras.summary()

# Convert model to CoreML
birnn_coreml = ct.convert(birnn_keras)
mlmodel_path = "TorchScriptVSCoreML/TorchScriptVSCoreML/MLModel/birnn.mlmodel"
birnn_coreml.save(mlmodel_path)

# Rename input/output
spec = ct.utils.load_spec(mlmodel_path)
ct.utils.rename_feature(spec, "bidirectional_input", "melspec")
ct.utils.rename_feature(spec, "Identity", "activations")
ct.utils.save_spec(spec, mlmodel_path)
