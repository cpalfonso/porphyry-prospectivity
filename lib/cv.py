"""Functions to perform cross-validation of models."""
import time
from sys import stderr
from typing import (
    Any,
    Literal,
    Mapping,
    Optional,
)

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import (
    ClassifierMixin,
    clone,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator

from .misc import (
    _PathOrDataFrame,
)
from .pu import get_xy

_METRICS = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "balanced_accuracy": balanced_accuracy_score,
    "accuracy": accuracy_score,
    "f1": f1_score,
}
_PROB_METRICS = {
    "roc_auc",
    "average_precision",
}


def perform_cv(
    clf: ClassifierMixin,
    data: _PathOrDataFrame,
    cv: Optional[BaseCrossValidator] = None,
    thresh: float = 0.5,
    random_state: Optional[int] = None,
    pu: bool = True,
    separate_regions: bool = True,
    stratify: Optional[ArrayLike] = None,
    verbose: bool = False,
    get_xy_kw: Optional[Mapping[str, Any]] = None,
    return_models: bool = False,
    thresh_method: Optional[
        Literal[
            "balanced_accuracy",
            "accuracy",
            "f1",
        ]
    ] = None,
    label: str = "label",
) -> pd.DataFrame:
    """Perform cross-validation.

    Parameters
    ----------
    clf : classification estimator
    data : str or DataFrame
        The training/test data to use.
    cv : cross-validator object, optional
        Scikit-learn cross-validator to use. If none is provided,
        a shuffled StratifiedKFold with 5 splits will be used.
    thresh : float or "auto", default: 0.5
        Probability threshold for classification. "auto" means
        the optimal threshold will be determined automatically, using the
        method specified in `thresh_method`.
    random_state : int, optional
        Seed for random number generation.
    pu : bool, default: True
        Whether `clf` is a positive-unlabelled classifier.
    separate_regions : bool, default: True
        Calculate test scores on different regions separately.
    stratify : array_like or str, optional
        The array or column name to use for stratification. By default,
        `data[label]` will be used.
    verbose : bool, default: False
        Print log to stderr.
    get_xy_kw : dict, optional
        Keyword arguments to be passed to `lib.pu.get_xy`.
    return_models : bool, default: False
        Include trained model objects in output.
    thresh_method : {"balanced_accuracy", "accuracy", "f1"}, default: "f1"
        Method used to determine optimal probability threshold if
        `thresh == "auto"`.
    label : str, default: "label"
        Label column of `data`.

    Returns
    -------
    DataFrame
        Performance metrics and statistics for each cross-validation split.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        data = pd.DataFrame(data)

    if cv is None:
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )
    n_splits = cv.get_n_splits()

    if stratify is None:
        stratify = data[label]
    elif stratify in data.columns:
        stratify = data[stratify]
    elif np.size(stratify) != data.shape[0]:
        raise ValueError(
            f"Invalid `stratify` parameter: {str(stratify)}"
        )
    split = cv.split(data, stratify)

    results = []
    for i, (train_idx, test_idx) in enumerate(split):
        if verbose:
            print(
                f"Current fold: {i + 1} of {n_splits}",
                file=stderr,
            )
        func = _region_cv if separate_regions else _combined_cv
        results.append(
            func(
                clf=clf,
                train_idx=train_idx,
                test_idx=test_idx,
                data=data,
                thresh=thresh,
                pu=pu,
                verbose=verbose,
                get_xy_kw=get_xy_kw,
                return_models=return_models,
                thresh_method=thresh_method,
                label=label,
            )
        )

    results = pd.concat(results, ignore_index=True)
    return results


def _region_cv(
    clf,
    train_idx,
    test_idx,
    data,
    thresh=0.5,
    pu=True,
    verbose=False,
    get_xy_kw=None,
    return_models=False,
    thresh_method=None,
    label="label",
):
    if get_xy_kw is None:
        get_xy_kw = {}
    if pu:
        labels = {"positive", "unlabeled", "unlabelled"}
    else:
        labels = {"positive", "negative"}

    if thresh == "auto":
        if thresh_method is None:
            thresh_method = "f1"
        if thresh_method not in {"balanced_accuracy", "accuracy", "f1"}:
            raise ValueError(f"Invalid thresh_method: {thresh_method}")
        auto_thresh = True
    else:
        auto_thresh = False

    regions = data["region"].unique()
    output = {i: [] for i in _METRICS.keys()}
    output["region"] = []
    for which in ("test", "train"):
        output[f"n_{which}"] = []
    for which in ("fit", "predict"):
        output[f"time_{which}"] = []
    if return_models:
        output["model"] = []
    if auto_thresh:
        output["prob_thresh"] = []

    train_data = data.loc[train_idx, :]
    train_data = train_data[train_data[label].isin(labels)]
    x_train, y_train = get_xy(train_data, **get_xy_kw)
    n_train = np.size(y_train)

    test_data = data.loc[test_idx, :]
    test_data = test_data[
        test_data[label].isin({"positive", "negative"})
    ]
    x_test, y_test = get_xy(test_data, **get_xy_kw)

    model = clone(clf)
    t0 = time.time()
    model.fit(x_train, y_train)
    fit_time = time.time() - t0

    probs = np.full_like(y_test, np.nan, dtype=np.float64)
    region_data = {}
    for region in regions:
        indices_region = np.where(test_data["region"] == region)[0]
        x_region = x_test[indices_region, :]
        t0 = time.time()
        probs_region = model.predict_proba(x_region)[:, 1]
        predict_time = time.time() - t0
        probs[indices_region] = probs_region
        region_data[region] = {
            "indices": indices_region,
            "time": predict_time,
        }
    region_data["All"] = {
        "indices": np.arange(np.size(probs)),
        "time": sum([region_data[region]["time"] for region in regions]),
    }

    if auto_thresh:
        test_thresholds = np.linspace(0, 1, 100)
        score_func = _METRICS[thresh_method]
        scores = np.zeros_like(test_thresholds)
        for thresh_i, test_threshold in enumerate(test_thresholds):
            scores[thresh_i] = score_func(y_test, probs >= test_threshold)
        thresh = np.median(test_thresholds[scores == np.nanmax(scores)])
    preds = (probs >= thresh).astype(np.int_)

    for region in region_data.keys():
        indices_region = region_data[region]["indices"]
        n_region = np.size(indices_region)
        y_region = y_test[indices_region]
        probs_region = probs[indices_region]
        preds_region = preds[indices_region]

        for metric, function in _METRICS.items():
            arg = probs_region if metric in _PROB_METRICS else preds_region
            value = function(y_region, arg)
            output[metric].append(value)

        output["n_train"].append(n_train)
        output["n_test"].append(n_region)
        output["time_fit"].append(fit_time)
        output["time_predict"].append(region_data[region]["time"])
        output["region"].append(region)
        if return_models:
            output["model"].append(model)
        if auto_thresh:
            output["prob_thresh"].append(thresh)

    output = pd.DataFrame(output)
    return output


def _combined_cv(
    clf,
    train_idx,
    test_idx,
    data,
    thresh=0.5,
    pu=True,
    verbose=False,
    get_xy_kw=None,
    return_models=False,
    thresh_method=None,
    label="label",
):
    if get_xy_kw is None:
        get_xy_kw = {}
    if pu:
        labels = {"positive", "unlabeled", "unlabelled"}
    else:
        labels = {"positive", "negative"}

    if thresh == "auto":
        if thresh_method is None:
            thresh_method = "f1"
        if thresh_method not in {"balanced_accuracy", "accuracy", "f1"}:
            raise ValueError(f"Invalid thresh_method: {thresh_method}")
        auto_thresh = True
    else:
        auto_thresh = False

    output = {i: [] for i in _METRICS.keys()}
    for which in ("test", "train"):
        output[f"n_{which}"] = []
    for which in ("fit", "predict"):
        output[f"time_{which}"] = []
    if return_models:
        output["model"] = []
    if auto_thresh:
        output["prob_thresh"] = []

    train_data = data.loc[train_idx, :]
    train_data = train_data[train_data[label].isin(labels)]
    x_train, y_train = get_xy(train_data, **get_xy_kw)
    n_train = np.size(y_train)

    test_data = data.loc[test_idx, :]
    test_data = test_data[
        test_data[label].isin({"positive", "negative"})
    ]
    x_test, y_test = get_xy(test_data, **get_xy_kw)
    n_test = np.size(y_test)

    model = clone(clf)
    t0 = time.time()
    model.fit(x_train, y_train)
    fit_time = time.time() - t0

    # probs = np.full_like(y_test, np.nan, dtype=np.float64)
    t0 = time.time()
    probs = model.predict_proba(x_test)[:, 1]
    predict_time = time.time() - t0

    if auto_thresh:
        test_thresholds = np.linspace(0, 1, 100)
        score_func = _METRICS[thresh_method]
        scores = np.zeros_like(test_thresholds)
        for thresh_i, test_threshold in enumerate(test_thresholds):
            scores[thresh_i] = score_func(y_test, probs >= test_threshold)
        thresh = np.median(test_thresholds[scores == np.nanmax(scores)])
    preds = (probs >= thresh).astype(np.int_)

    output = {
        "n_train": [n_train],
        "n_test": [n_test],
        "time_fit": [fit_time],
        "time_predict": [predict_time],
    }
    if return_models:
        output["model"] = [model]
    if auto_thresh:
        output["prob_thresh"] = [thresh]
    for metric, function in _METRICS.items():
        arg = probs if metric in _PROB_METRICS else preds
        value = function(y_test, arg)
        output[metric] = [value]

    output = pd.DataFrame(output)
    return output
