from beta_nmf_minibatch import BetaNMF
from base import nnrandn
import numpy as np

from bokeh.plotting import figure, show
from bokeh.io import output_file

n_iter = 100
n_components = 10
beta = 2
batch_size = 100
cache1_size = 500
sag_mem = 2
X = nnrandn((500, 20))
H_init = nnrandn((X.shape[0], n_components))
W_init = nnrandn((X.shape[1], n_components))

nmf = BetaNMF(X.shape, n_components, beta, n_iter, verbose=10,
              cache1_size=cache1_size, batch_size=batch_size,
              W=W_init, H=H_init, init_mode='custom', solver='mu_batch')
score_mu_batch = nmf.fit(X)

nmf = BetaNMF(X.shape, n_components, beta, n_iter, verbose=10,
              cache1_size=cache1_size, batch_size=batch_size,
              W=W_init, H=H_init, init_mode='custom', solver='asg_mu')
score_asg_mu = nmf.fit(X)

nmf = BetaNMF(X.shape, n_components, beta, n_iter, verbose=10,
              cache1_size=cache1_size, batch_size=batch_size,
              W=W_init, H=H_init, init_mode='custom', solver='gsg_mu')
score_gsg_mu = nmf.fit(X)

nmf = BetaNMF(X.shape, n_components, beta, n_iter, verbose=10,
              cache1_size=cache1_size, batch_size=batch_size,
              W=W_init, H=H_init, init_mode='custom', solver='asag_mu')
score_asag_mu = nmf.fit(X)

nmf = BetaNMF(X.shape, n_components, beta, n_iter, verbose=10,
              cache1_size=cache1_size, batch_size=batch_size,
              W=W_init, H=H_init, init_mode='custom', solver='gsag_mu')
score_gsag_mu = nmf.fit(X)

output_file(f'{__file__}.html')

p = figure(title="Mini-batch algorithms performance (cost)")

#MU mini-batch
p.line(score_mu_batch[1:, 1], score_mu_batch[1:, 0], 
       legend_label="MU mini batch",
       line_dash=[4, 4], line_color="red", line_width=2)
#ASG-MU
p.square(score_asg_mu[1:, 1], score_asg_mu[1:, 0],
          legend_label="ASG-MU",
          fill_color="green", line_color="black")
p.line(score_asg_mu[1:, 1], score_asg_mu[1:, 0],
        legend_label="ASG-MU",
        line_color="green", line_width=2)
#GSG_MU
p.square(score_gsg_mu[2:, 1], score_gsg_mu[2:, 0],
          legend_label="GSG-MU",
          fill_color="green", line_color="black")
p.line(score_gsg_mu[1:, 1], score_gsg_mu[1:, 0],
        legend_label="GSG-MU",
        line_color="green")
#ASAG-MU
p.circle(score_asag_mu[3:, 1], score_asag_mu[3:, 0],
          legend_label="ASAG-MU",
          fill_color="black", line_color="black")
p.line(score_asag_mu[1:, 1], score_asag_mu[1:, 0],
        legend_label="ASAG-MU",
        line_color="black", line_width=2)
#GSAG_MU
p.circle(score_gsag_mu[4:, 1], score_gsag_mu[4:, 0],
          legend_label="GSAG-MU",
          fill_color="black", line_color="black")
p.line(score_gsag_mu[1:, 1], score_gsag_mu[1:, 0],
        legend_label="GSAG-MU",
        line_color="black")

show(p)
