
def get_true_values(N):
    df = pd.read_csv('/home/cdac-c-15/PycharmProjects/EstimatePK_PD/simulation_data/population_parameters.csv')
    df = df[(df.ID.between(1, N))]
    df = df.drop_duplicates()
    df_patient_true = df[['A_true', 'c_true', 'gamma_true', 'Age', 'SOFA']].rename(columns={
        'A_true': 'A', 'c_true': 'c', 'gamma_true': 'gamma'
    })
    return df_patient_true


def gen_sum_values(ci_table, true_values):

    lower = np.asarray(ci_table.loc[.025]) # shape: (N, )
    middle = np.asarray(ci_table.loc[0.50])
    upper = np.asarray(ci_table.loc[.975]) # shape: (N, )
    within_interval, coverage_rate = check_if_match(true_values, lower, upper)

    return lower, middle, upper, within_interval, coverage_rate


def check_if_match(true_params, lower, upper):
    """
    true_params: shape = (N,), the true param value for each patient
    #check whether the true parameter lies within the interval is a good way to evaluate model accuracy on simulated data.
    this measures how well the model can estimate the parameter
    """
    within_interval = (lower <= true_params) & (true_params <= upper)
    coverage_rate = np.mean(within_interval)
    return within_interval, coverage_rate

def init_fn():
    return {
        "beta_A_0": np.random.normal(0.97, 0.05),           # around typical logit intercept
        "beta_A_age": np.random.normal(0.0, 5e-4),         # centered at 0, wide enough for slope
        "beta_A_SOFA": np.random.normal(0.0, 5e-4),

        "beta_c_0": np.random.normal(2.0, 0.5),            # log scale baseline ~ exp(2) = ~7.4
        "beta_c_age": np.random.normal(0.0, 0.01),
        "beta_c_SOFA": np.random.normal(0.0, 0.01),

        "beta_gamma_0": np.random.normal(1.5, 0.5),        # log scale ~ exp(1.5) = ~4.5
        "beta_gamma_age": np.random.normal(0.0, 0.01),
        "beta_gamma_SOFA": np.random.normal(0.0, 0.01)
    }
def prepare_sim_data(N,T, sim_data_path):
    df = pd.read_csv(sim_data_path)
    dt = 1 / 60
    start_time = 3.983
    end_time = start_time + dt * T
    # Filter to first N patients
    df_filtered = df[
        (df['SubjectID'].between(1, N)) & (df['Time_h'] <= end_time) & (df['Time_h'] > start_time) & (df['Time_h'] != 0)]
    # Update subject list based on filtered data
    subject_ids = df_filtered["SubjectID"].unique()
    # Initialize matrices
    fp = np.zeros((N, T))
    u = np.zeros((N, T))
    event = np.zeros((N, T), dtype=int)

    for i, sid in enumerate(subject_ids):
        sub = df_filtered[df_filtered["SubjectID"] == sid].sort_values("Time_h")
        fp[i, :] = sub["fp"].values
        u[i, :] = sub["u"].values
        event[i, :] = sub["event"].values
    fp_not_log = np.exp(fp)
    return fp_not_log, u, event

def check_if_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def patient_summary_helper(df_true):
    A_true = df_true['A'].values
    c_true = df_true['c'].values * drug_harm_scale
    gamma_true = df_true['gamma'].values * gamma_scale
    return A_true, c_true, gamma_true


def save_patient_summary(df_fit, df_true, summary, N, T, run_descrptn, output_directory):
    params = ['A', 'c', 'gamma']
    patient_param_summaries = {}
    ci_table = {}
    A_true, c_true, gamma_true = patient_summary_helper(df_true)
    for p in params:
        # Switch-style logic for true values
        if p == 'A':
            true_values = A_true
        elif p == 'c':
            true_values = c_true
        elif p == 'gamma':
            true_values = gamma_true
        else:
            raise ValueError(f"Unknown parameter: {p}")
        rhat_series = summary.loc[summary.index.str.startswith(p + '['), 'R_hat']
        cols = [col for col in df_fit.columns if col.startswith(p + '[')]
        ci_table[p] = df_fit[cols].quantile([0.025, 0.50, 0.975])
        lower, middle, upper, within, coverage = gen_sum_values(ci_table[p], true_values)
        error = np.abs(true_values - middle)

        patient_param_summaries[p] = pd.DataFrame({
            'Patient': np.arange(1, N + 1),
            'true': true_values,
            'estimate': middle,
            'AE': error,
            'lower': lower,
            'upper': upper,
            'w_in_interval': within,
            'r_hat': rhat_series.values
        })
        print(f"MAE {p}:", np.mean(error))

    patient_summary = pd.concat(patient_param_summaries, names=['parameter'])
    patient_summary.to_csv(f"{output_directory}/{N}_pats_{T}_points_PATIENT_{run_descrptn}.csv")
    return patient_summary

def save_population_summary(df_fit, summary, N, T, run_descrptn, output_directory):
    df_pop_true = pd.read_csv('/home/cdac-c-15/PycharmProjects/EstimatePK_PD/Flattened_Coefficients.csv')

    param_names = [
        "beta_c_0", "beta_c_age", "beta_c_SOFA",
        "beta_gamma_0", "beta_gamma_age", "beta_gamma_SOFA",
        "beta_A_0", "beta_A_age", "beta_A_SOFA"
    ]

    pop_param_summaries = []

    for p in param_names:
        samples = df_fit[p]
        lower = samples.quantile(0.025)
        middle = samples.quantile(0.50)
        upper = samples.quantile(0.975)
        true_val = df_pop_true[p].values[0]
        error = np.abs(true_val - middle)
        w_in_interval = lower <= true_val <= upper
        rhat_val = summary.loc[p, "R_hat"]
        pop_param_summaries.append({
            "parameter": p,
            "true": true_val,
            "estimate": middle,
            "AE": error,
            "lower": lower,
            "upper": upper,
            "w_in_interval": w_in_interval,
            "r_hat": rhat_val
        })
        print(f"MAE {p}:", np.mean(error))

    pop_summary = pd.DataFrame(pop_param_summaries)
    pop_summary.to_csv(f"{output_directory}/{N}_pats_{T}_points_POP_{run_descrptn}.csv")
    return pop_summary

from cmdstanpy import CmdStanModel
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import arviz as az


os.environ["STAN_NUM_THREADS"] = "8"
# ====== scenarios ========
run_name = sys.argv[1]
sim_data_path = sys.argv[2]
num_patients = int(sys.argv[3])
out_directory_path = sys.argv[4]

drug_harm_scale = float(sys.argv[5])
gamma_scale = float(sys.argv[6])
ea_burden_scale = float(sys.argv[7])
print("everything read in")

# Settings
N = num_patients # number of patients
T = int(122)
run_descrptn = run_name

#  === get true params ===
df_patient_true = get_true_values(N)
mean_age = np.mean(df_patient_true['Age'])
mean_SOFA = np.mean(df_patient_true['SOFA'])
age_ctrd = df_patient_true['Age'].values - mean_age
SOFA_ctrd = df_patient_true['SOFA'].values - mean_SOFA

fp_not_log, u, event = prepare_sim_data(N,T,sim_data_path)
print("simulation data retrieved successfully")
prior_sigma_0 = 1.5 * 1.5
prior_sigma_age = 1.5 * 1.5
prior_sigma_SOFA = 1.5 * 1.5

# Package data for Stan
stan_data = {
    "N": N,
    "T": T,
    "fp_not_log": fp_not_log,
    "u": u,
    "event": event,
    "dt": 1 / 60,
    "age_ctrd": age_ctrd,
    "SOFA_ctrd": SOFA_ctrd,
    "prior_sigma_0": prior_sigma_0,
    "prior_sigma_age": prior_sigma_age,
    "prior_sigma_SOFA": prior_sigma_SOFA
}
print("STAN_NUM_THREADS =", os.environ.get("STAN_NUM_THREADS"))
# Compile and sample
model = CmdStanModel(stan_file='/home/cdac-c-15/PycharmProjects/a2_pk_pd_threaded/log_scale_c_gamma.stan', cpp_options={'STAN_THREADS': True}, force_compile=True)
fit = model.sample(data=stan_data,
                   chains=1,
                   threads_per_chain=8,
                   iter_sampling=1000,
                   iter_warmup=1000,
                   show_console=True,
                   adapt_delta=0.7,
                   inits=init_fn())
df_fit = fit.draws_pd()
summary = fit.summary()

output_directory = Path(out_directory_path) / run_descrptn
check_if_exists(output_directory)

save_patient_summary(df_fit, df_patient_true, summary, N, T, run_descrptn, output_directory)
save_population_summary(df_fit, summary, N, T, run_descrptn, output_directory)
idata = az.from_cmdstanpy(posterior=fit)  # or from_dict/from_numpy depending on how you build it
draws_to_save = idata.to_netcdf(f"{output_directory}/{run_descrptn}.nc")
