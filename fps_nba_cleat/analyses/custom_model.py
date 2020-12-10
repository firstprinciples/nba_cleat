import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import multivariate_normal
tfd = tfp.distributions

class GameCell(tf.keras.layers.Layer):
    
    def __init__(self, states, teams, lines_len, scores_len, l1=0, l2=0):
        super(GameCell, self).__init__()
        self.states = states
        self.teams = teams
        self.lines_len = lines_len
        self.scores_len = scores_len
        self.l1 = l1
        self.l2 = l2
        self.state_size = tf.TensorShape([teams, states])

    def build(self, input_shape ):
        self.Bz = self.add_weight(
            name='Bz',
            shape=[self.states*2, 4 + self.lines_len + self.scores_len],
            regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2))
        self.Br = self.add_weight(
            name='Br',
            shape=[self.states*2, 4 + self.lines_len + self.scores_len],
            regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2))
        self.Bm = self.add_weight(
            name='Bm',
            shape=[self.states*2, 4 + self.lines_len + self.scores_len],
            regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2))
        self.Az = self.add_weight(
            name='Az',
            shape=[self.states*2, self.states*2], 
            # initializer=tf.keras.initializers.Identity(gain=0.5),
            regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2))
        self.Ar = self.add_weight(
            name='Ar',
            shape=[self.states*2, self.states*2],
            regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2))
            # initializer=tf.keras.initializers.Identity(gain=0.25))
        self.Am = self.add_weight(
            name='Am',
            shape=[self.states*2, self.states*2],
            regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2))
            # initializer=tf.keras.initializers.Identity(gain=0.25))
        self.dz = self.add_weight(
            name='dz',
            shape=[self.states*2, 1],)
            # initializer=tf.keras.initializers.Zeros)
        self.dr = self.add_weight(
            name='dr',
            shape=[self.states*2, 1],
            initializer=tf.keras.initializers.Ones())
        self.dm = self.add_weight(
            name='dm',
            shape=[self.states*2, 1],)
            # initializer=tf.keras.initializers.Zeros)
        super(GameCell, self).build(input_shape)
    
    def call(self, inputs, state):
        teams, u = inputs
        u = tf.reshape(u, (-1, 1))
        state = state[0]
        x_k = self._get_state(state, teams)
        x_kp = self._gru(u, x_k)
        state = self._update_state(state, teams, x_kp, x_k)
        state = tf.reshape(state, (1, state.shape[0], state.shape[1]))
        return state, [state]

    def _gru(self, u, x):
        z = self.Az @ x + self.Bz @ u + self.dz
        z = tf.math.sigmoid(z)
        r = self.Ar @ x + self.Br @ u - self.dr
        r = tf.math.sigmoid(r)
        m = self.Am @ (r * x) + self.Bm @ u + self.dm
        m = tf.math.tanh(m)
        return z * x + (1-z) * m

    def _linear(self, u, x):
        return self.Az @ x + self.Bz @ u + self.dz

    @staticmethod
    def _get_state(state, teams):
        states = tf.gather_nd(state[0], teams)
        return tf.reshape(states, (-1, 1))
    
    @staticmethod
    def _update_state(state, teams, x_kp, x_k):
        dx = x_kp - x_k
        dx = tf.reshape(dx, (2, -1))
        indices = tf.reshape(teams, (2, 1))
        return state[0] + tf.scatter_nd(indices, dx, state[0].shape)

class GameRNN(tf.keras.layers.RNN):
    
    def __init__(self, states, teams, lines_len, scores_len, 
                 l1, l2, return_sequences=True, return_state=False,
                 stateful=True, unroll=False):
        cell = GameCell(states, teams, lines_len, scores_len, l1, l2)
        super(GameRNN, self).__init__(cell, 
            return_sequences=return_sequences, return_state=return_state,
            stateful=stateful, unroll=unroll)
        
    def call(self, inputs, initial_state=None, constants=None):
        return super(GameRNN, self).call(
            inputs, initial_state=initial_state, constants=constants)

class PredictionLayer(tf.keras.layers.Layer):
    
    def __init__(self, states, l1=0, l2=0, predictions=2):
        super(PredictionLayer, self).__init__()
        self.states = states
        self.l1 = l1
        self.l2 = l2
        self.predictions = predictions

    def build(self, input_shape):
        self.C = self.add_weight(
            name='C',
            shape=[self.predictions, self.states*2],
            regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2))
        self.D = self.add_weight(
            name='D',
            shape=[self.predictions, 6],
            regularizer=tf.keras.regularizers.l1_l2(self.l1, self.l2))
        self.const = self.add_weight(
            name='const',
            shape=[self.predictions, 1])
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs):
        u_kp, state, teams = inputs
        u_kp = tf.expand_dims(u_kp, axis=-1)[0]
        x_kp = self._get_state(state, teams)
        Cx = tf.transpose(tf.tensordot(self.C, x_kp, axes=[[1], [1]]), perm=[1, 0, 2])
        Du = tf.transpose(tf.tensordot(self.D, u_kp, axes=[[1], [1]]), perm=[1, 0, 2])
        return tf.expand_dims(Cx + Du + self.const, axis=0)

    @staticmethod
    def _get_state(state, teams):
        seq_len = teams.shape[1]
        seq_len_range = tf.expand_dims(
            tf.range(seq_len, dtype=tf.int32), axis=1)
        home_indices = tf.concat((seq_len_range, teams[0, :, 0]), axis=1)
        visitor_indices = tf.concat((seq_len_range, teams[0, :, 1]), axis=1)
        state_home = tf.gather_nd(state[0], home_indices)
        state_visitor = tf.gather_nd(state[0], visitor_indices)
        state_teams = tf.concat((state_home, state_visitor), axis=1)
        return tf.expand_dims(state_teams, axis=-1)

def build_model(seq_len, states, teams, l1=0, l2=0, lines_len=4, scores_len=8):
    
    teams_k = tf.keras.layers.Input(batch_shape=(1, seq_len, 2, 1), dtype=tf.int32)
    days_off_k = tf.keras.layers.Input(batch_shape=(1, seq_len, 4))
    lines_k = tf.keras.layers.Input(batch_shape=(1, seq_len, lines_len))
    scores_k = tf.keras.layers.Input(batch_shape=(1, seq_len, scores_len))

    u_k = tf.keras.layers.concatenate([days_off_k, lines_k, scores_k])

    x_kp = GameRNN(states, teams, lines_len, scores_len, l1=l1, l2=l2)(tuple([teams_k, u_k]))

    teams_kp = tf.keras.layers.Input(batch_shape=(1, seq_len, 2, 1), dtype=tf.int32)
    days_off_kp = tf.keras.layers.Input(batch_shape=(1, seq_len, 4))
    lines_kp = tf.keras.layers.Input(batch_shape=(1, seq_len, 2))

    u_kp = tf.keras.layers.concatenate([days_off_kp, lines_kp])

    preds = PredictionLayer(states, l1=l1, l2=l2, predictions=5)([u_kp[..., :], x_kp, teams_kp])
    preds = tf.keras.layers.Reshape((seq_len, -1))(preds)

    score_dist = tfp.layers.DistributionLambda(
        lambda t: tfd.MultivariateNormalTriL(
            loc=t[..., :2], 
            scale_tril=tfp.math.fill_triangular(
                1e-3 + tf.math.softplus(t[..., 2:])
            )
        )
    )(preds)

    return tf.keras.Model(
        inputs=[teams_k, days_off_k, lines_k, scores_k, teams_kp, days_off_kp, lines_kp],
        outputs=[score_dist])

class ResultsProcessor:
    
    N_GRID = 101#31

    def __init__(self, score_scaler, min_score=60, max_score=160):

        self.score_scaler = score_scaler
        self.score_grid = np.linspace(min_score, max_score, self.N_GRID)
        self.grid = np.meshgrid(self.score_grid, self.score_grid)
        self.grid = np.moveaxis(np.stack(self.grid), source=0, destination=2)
        self.grid = self.grid.reshape(-1, 2)
        self.scaled_grid = self.score_scaler.transform(
            self.grid.reshape(-1, 1)).reshape(-1, 2)
        self.spread_grid = self.grid[..., 0] - self.grid[..., 1]
        self.total_grid = self.grid[..., 0] + self.grid[..., 1]

    def get_spread_total_likelihood(self, mvns, spreads, totals):
        spread_beat = []
        total_beat = []
        for i in range(len(spreads)):
            spread_beat_grid = np.where(self.spread_grid > spreads[i])
            total_beat_grid = np.where(self.total_grid > totals[i])
            likelihood_spread = np.sum(mvns[i].pdf(self.scaled_grid[spread_beat_grid]))
            likelihood_spread /= np.sum(self._get_density(mvns[i]))
            likelihood_total = np.sum(mvns[i].pdf(self.scaled_grid[total_beat_grid]))
            likelihood_total /= np.sum(self._get_density(mvns[i]))
            spread_beat += [likelihood_spread]
            total_beat += [likelihood_total]

        return np.stack(spread_beat), np.stack(total_beat)

    def _get_density(self, mvn):
        return mvn.pdf(self.scaled_grid).reshape(-1, self.N_GRID)

    @staticmethod
    def make_mvn_from_batch(batch_results):
        mvns = []
        for i in range(batch_results.batch_shape[1]):
            mvns += [multivariate_normal(
                mean=batch_results.mean()[0, i].numpy(),
                cov=batch_results.covariance()[0, i].numpy()
            )]
        return mvns