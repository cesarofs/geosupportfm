import numpy as np
import pandas as pd

from geosupportfm.embeddings.analysis import (
    EvaluationResult,
    axis_importance,
    centroid_cosine_similarity,
    class_centroids,
    count_samples_by_year_class,
    evaluate_classifier,
    feature_matrix,
    fit_random_forest,
    label_vector,
    prepare_dataframe,
    split_by_year,
    top_similar_pairs,
)


def make_frame():
    return pd.DataFrame(
        {
            "A00": [1.0, 1.1, 3.0, 3.2],
            "A01": [0.1, 0.2, 0.9, 1.0],
            "lulc": [1, 1, 2, 2],
            "year": [2020, 2020, 2024, 2024],
        }
    )


def test_prepare_dataframe_casts_types_and_drops_nulls():
    df = make_frame()
    df.loc[0, "A00"] = np.nan
    cleaned = prepare_dataframe(df, ["A00", "A01"], "lulc", "year")
    assert cleaned.shape[0] == 3
    assert cleaned["lulc"].dtype.kind in {"i", "u"}
    assert cleaned["year"].dtype.kind in {"i", "u"}


def test_split_by_year():
    df = make_frame()
    train, test = split_by_year(df, [2020], [2024])
    assert len(train) == 2
    assert len(test) == 2


def test_feature_and_label_extraction():
    df = make_frame()
    x = feature_matrix(df, ["A00", "A01"])
    y = label_vector(df)
    assert x.shape == (4, 2)
    assert y.tolist() == [1, 1, 2, 2]


def test_fit_and_evaluate_classifier():
    df = make_frame()
    train = df.iloc[[0, 2]].copy()
    test = df.iloc[[1, 3]].copy()
    x_train = feature_matrix(train, ["A00", "A01"])
    y_train = label_vector(train)
    x_test = feature_matrix(test, ["A00", "A01"])
    y_test = label_vector(test)

    model = fit_random_forest(x_train, y_train, seed=0, n_estimators=50)
    result, pred = evaluate_classifier(model, x_test, y_test)

    assert isinstance(result, EvaluationResult)
    assert pred.shape == y_test.shape
    assert 0.0 <= result.balanced_accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0


def test_axis_importance_and_class_statistics():
    df = make_frame()
    train, _ = split_by_year(df, [2020], [2024])
    x_train = feature_matrix(train, ["A00", "A01"])
    y_train = label_vector(train)
    model = fit_random_forest(x_train, y_train, seed=0, n_estimators=20)

    importance = axis_importance(model, ["A00", "A01"])
    assert list(importance.index) == ["A00", "A01"] or list(importance.index) == ["A01", "A00"]

    centroids = class_centroids(df, ["A00", "A01"])
    similarity = centroid_cosine_similarity(centroids)
    assert similarity.shape == (2, 2)

    pairs = top_similar_pairs(similarity, top_n=1)
    assert len(pairs) == 1
    assert pairs[0][2] <= 1.0


def test_count_samples_by_year_class():
    df = make_frame()
    counts = count_samples_by_year_class(df)
    assert counts["n"].sum() == 4
