functions {
  real partial_sum_lpmf(array[,] int event_slice,
                        int start, int end,
                        matrix fp_not_log,
                        matrix u,
                        vector age_ctrd,
                        vector SOFA_ctrd,
                        real beta_A_0,
                        real beta_A_age,
                        real beta_A_SOFA,
                        vector A_raw,
                        real beta_c_0,
                        real beta_c_age,
                        real beta_c_SOFA,
                        vector c_raw,
                        real beta_gamma_0,
                        real beta_gamma_age,
                        real beta_gamma_SOFA,
                        vector gamma_raw,
                        real dt,
                        int T) {

    real lp = 0;

    for (s in 1:(end - start + 1)) {
      int i = start + s - 1;
      real reg_A = beta_A_0 + beta_A_age * age_ctrd[i] + beta_A_SOFA * SOFA_ctrd[i] + 0.012375 * A_raw[i];
      real A = fmin(reg_A, 1 - 1e-6);
      real c = beta_c_0 + beta_c_age * age_ctrd[i] + beta_c_SOFA * SOFA_ctrd[i] + 0.2 * c_raw[i];
      real gamma = beta_gamma_0 + beta_gamma_age * age_ctrd[i] + beta_gamma_SOFA * SOFA_ctrd[i] + 0.15 * gamma_raw[i];

      vector[T] x;
      vector[T] logit_p;

      x[1] = u[i, 1];
      for (t in 2:T) {
        x[t] = A * x[t - 1] + u[i, t];
      }

      for (t in 1:T) {
        real x_safe = x[t] + 1e-6;
        real log_r = gamma * (log(c) - log(x_safe));
        real g = 1 - inv_logit(-log_r);
        real lambda = fp_not_log[i, t] * g;
        real prob = fmin(fmax(lambda * dt, 1e-9), 1 - 1e-9);
        logit_p[t] = log(prob / (1 - prob));
      }

      lp += bernoulli_logit_lpmf(event_slice[s] | logit_p);
    }

    return lp;
  }
}

data {
  int<lower=1> N;
  int<lower=1> T;
  matrix[N, T] fp_not_log;
  matrix[N, T] u;
  array[N, T] int<lower=0, upper=1> event;
  real<lower=0> dt;
  vector[N] age_ctrd;
  vector[N] SOFA_ctrd;
  real prior_sigma_0;
  real prior_sigma_age;
  real prior_sigma_SOFA;
}

parameters {
  real beta_A_0;
  real beta_A_age;
  real beta_A_SOFA;
  vector[N] A_raw;

  real beta_c_0;
  real beta_c_age;
  real beta_c_SOFA;
  vector[N] c_raw;

  real beta_gamma_0;
  real beta_gamma_age;
  real beta_gamma_SOFA;
  vector[N] gamma_raw;
}

model {
  beta_A_0 ~ normal(.9, 0.1 * prior_sigma_0);
  beta_A_age ~ normal(0, 0.001);
  beta_A_SOFA ~ normal(0, 0.001);

  beta_c_0 ~ normal(3, 1);
  beta_c_age ~ normal(0, .01);
  beta_c_SOFA ~ normal(0, 0.01);

  beta_gamma_0 ~ normal(1.5, .5);
  beta_gamma_age ~ normal(0, 0.01);
  beta_gamma_SOFA ~ normal(0, 0.01);

  gamma_raw ~ normal(0, 1);
  c_raw ~ normal(0, 1);
  A_raw ~ normal(0, 1);

  target += reduce_sum(
    partial_sum_lpmf,
    event,
    10,
    fp_not_log, u,
    age_ctrd, SOFA_ctrd,
    beta_A_0, beta_A_age, beta_A_SOFA, A_raw,
    beta_c_0, beta_c_age, beta_c_SOFA, c_raw,
    beta_gamma_0, beta_gamma_age, beta_gamma_SOFA, gamma_raw,
    dt,
    T
  );
}

generated quantities {
  vector[N] A;
  vector[N] c;
  vector[N] gamma;
  array[N, T] real logit_p;
  array[N, T] real concentration;

  for (i in 1:N) {
    A[i] = beta_A_0 + beta_A_age * age_ctrd[i] + beta_A_SOFA * SOFA_ctrd[i] + 0.012375 * A_raw[i];
    c[i] = beta_c_0 + beta_c_age * age_ctrd[i] + beta_c_SOFA * SOFA_ctrd[i] + 0.2 * c_raw[i];
    gamma[i] = beta_gamma_0 + beta_gamma_age * age_ctrd[i] + beta_gamma_SOFA * SOFA_ctrd[i] + 0.15 * gamma_raw[i];

    vector[T] x;
    x[1] = u[i, 1];
    for (t in 2:T) {
      x[t] = fmin(A[i], 1 - 1e-6) * x[t - 1] + u[i, t];
    }

    for (t in 1:T) {
      real x_safe = x[t] + 1e-6;
      real log_r = gamma[i] * (log(c[i]) - log(x_safe));
      real g = 1 - inv_logit(-log_r);
      real lambda = fp_not_log[i, t] * g;
      real prob = fmin(fmax(lambda * dt, 1e-9), 1 - 1e-9);
      logit_p[i, t] = log(prob / (1 - prob));
      concentration[i, t] = x[t];
    }
  }
}

