from src.model import BioLeakyRNN, get_activation
from src.env import CuedTargetWithDistractorsV3, SpatialPretrain, CuedTargetSpatialV3
from src.training import TrainConfig, train_supervised
from src.dataset import rollout_one_trial, make_train_batch
from src.analysis import (
    collect_trials, filter_trials, select_trials,
    fit_pca_on_trials, get_aligned_pca_segments,
    dpca_marginals, collect_aligned_hidden_by_label, make_condition_mean_tensor,
    extract_window_features,
    prepare_jpca_input, fit_jpca, jpca_permutation_test, jpca_permutation_test_condition_shuffle,
    compute_tangling, tangling_by_ctoa_bin, polynomial_regression,
    decode_position_by_ctoa_bin,
)
