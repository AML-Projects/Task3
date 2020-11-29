import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_slice, plot_parallel_coordinate, plot_contour

if __name__ == '__main__':
    peak_detection_name = "wfdb"

    study = optuna.create_study(study_name="xgboost",
                                direction='maximize',
                                storage='sqlite:///optuna/search_' + peak_detection_name + '.db',
                                # n_startup_trials: number of random searches before TPE starts
                                sampler=TPESampler(n_startup_trials=20),
                                load_if_exists=True)

    print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))
    hist = study.trials_dataframe()
    hist.to_csv("./optuna/study_" + peak_detection_name + ".csv")

    plot_optimization_history(study).show()
    plot_slice(study).show()
    plot_parallel_coordinate(study).show()
    plot_contour(study).show()
