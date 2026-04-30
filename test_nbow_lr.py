"""
test_nbow_lr.py
---------------
Tests for src/nbow_lr.py — Tuqa's NBOW + Logistic Regression pipeline.

Run from the project root:
    PYTHONPATH=. python tests/test_nbow_lr.py
    PYTHONPATH=. python -m pytest tests/test_nbow_lr.py -v

No Jigsaw CSV needed — all tests use synthetic data.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("..")
from src.nbow_lr import (
    build_vectorizer,
    get_top_features,
    load_model,
    run_ablation,
    save_model,
    tune_threshold,
    train_lr,
    vectorize_splits,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

TOXIC_TEXTS = [
    "you are an idiot and i hate you",
    "shut up you stupid moron",
    "go die nobody likes you fool",
    "you are worthless garbage trash",
    "i will destroy you complete idiot",
    "disgusting filth you should be ashamed",
    "what a pathetic loser you are",
    "absolute scum of the earth moron",
]

CLEAN_TEXTS = [
    "i really enjoyed reading this article thanks",
    "could you clarify what you meant please",
    "this is a great point well made",
    "looking forward to the next update soon",
    "thanks for the helpful explanation really useful",
    "interesting perspective i had not considered that",
    "the analysis here is quite thorough well done",
    "nice work on this project very impressive",
]


def make_splits(
    n_toxic: int = 400,
    n_clean: int = 1600,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create synthetic train/val/test splits mimicking Jigsaw structure."""
    rng = np.random.default_rng(seed)

    all_texts  = (TOXIC_TEXTS * (n_toxic  // len(TOXIC_TEXTS) + 1))[:n_toxic]
    all_texts += (CLEAN_TEXTS * (n_clean  // len(CLEAN_TEXTS) + 1))[:n_clean]

    # Add slight variation so not all rows are identical
    varied = []
    for i, t in enumerate(all_texts):
        varied.append(t + f" word{i}")
    all_texts = varied

    labels = [1] * n_toxic + [0] * n_clean
    perm = rng.permutation(len(all_texts))
    texts  = [all_texts[i] for i in perm]
    labels = [labels[i]    for i in perm]

    df = pd.DataFrame({"id": range(len(texts)),
                       "comment_text": texts,
                       "clean_text":   texts,
                       "label":        labels})

    n = len(df)
    tr_end  = int(n * 0.8)
    val_end = int(n * 0.9)
    return (
        df.iloc[:tr_end].reset_index(drop=True),
        df.iloc[tr_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Tests: build_vectorizer
# ---------------------------------------------------------------------------

class TestBuildVectorizer(unittest.TestCase):

    def test_returns_tfidf_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = build_vectorizer()
        self.assertIsInstance(vec, TfidfVectorizer)

    def test_default_ngram_range(self):
        vec = build_vectorizer()
        self.assertEqual(vec.ngram_range, (1, 2))

    def test_custom_ngram_range(self):
        vec = build_vectorizer(ngram_range=(1, 1))
        self.assertEqual(vec.ngram_range, (1, 1))

    def test_default_max_features(self):
        vec = build_vectorizer()
        self.assertEqual(vec.max_features, 50_000)

    def test_custom_max_features(self):
        vec = build_vectorizer(max_features=1000)
        self.assertEqual(vec.max_features, 1000)

    def test_sublinear_tf_enabled(self):
        vec = build_vectorizer(sublinear_tf=True)
        self.assertTrue(vec.sublinear_tf)

    def test_sublinear_tf_disabled(self):
        vec = build_vectorizer(sublinear_tf=False)
        self.assertFalse(vec.sublinear_tf)

    def test_unfitted_before_transform(self):
        vec = build_vectorizer()
        with self.assertRaises(Exception):
            vec.transform(["hello world"])

    def test_fits_and_transforms(self):
        # Use enough documents so min_df=3 and max_df=0.95 do not conflict
        vec = build_vectorizer(max_features=100)
        texts = TOXIC_TEXTS + CLEAN_TEXTS   # 16 docs, terms repeat across them
        X = vec.fit_transform(texts)
        self.assertEqual(X.shape[0], len(texts))
        self.assertLessEqual(X.shape[1], 100)

    def test_min_df_respected(self):
        # With min_df=2, a term in only 1 doc should be dropped
        vec = build_vectorizer(max_features=10000, min_df=2, ngram_range=(1,1))
        texts = ["hello world", "hello there", "world peace unique_rare_word"]
        vec.fit(texts)
        vocab = vec.vocabulary_
        self.assertNotIn("unique_rare_word", vocab)
        self.assertIn("hello", vocab)


# ---------------------------------------------------------------------------
# Tests: vectorize_splits
# ---------------------------------------------------------------------------

class TestVectorizeSplits(unittest.TestCase):

    def setUp(self):
        self.train, self.val, self.test = make_splits()

    def test_returns_four_items(self):
        result = vectorize_splits(self.train, self.val, self.test)
        self.assertEqual(len(result), 4)

    def test_matrix_shapes(self):
        X_tr, X_va, X_te, vec = vectorize_splits(self.train, self.val, self.test)
        self.assertEqual(X_tr.shape[0], len(self.train))
        self.assertEqual(X_va.shape[0], len(self.val))
        self.assertEqual(X_te.shape[0], len(self.test))

    def test_all_same_feature_count(self):
        X_tr, X_va, X_te, _ = vectorize_splits(self.train, self.val, self.test)
        self.assertEqual(X_tr.shape[1], X_va.shape[1])
        self.assertEqual(X_tr.shape[1], X_te.shape[1])

    def test_vectorizer_returned_is_fitted(self):
        _, _, _, vec = vectorize_splits(self.train, self.val, self.test)
        self.assertIsNotNone(vec.vocabulary_)
        self.assertGreater(len(vec.vocabulary_), 0)

    def test_missing_text_col_raises(self):
        bad_train = self.train.drop(columns=["clean_text"])
        with self.assertRaises(ValueError):
            vectorize_splits(bad_train, self.val, self.test)

    def test_custom_text_col(self):
        train2 = self.train.rename(columns={"clean_text": "processed"})
        val2   = self.val.rename(columns={"clean_text": "processed"})
        test2  = self.test.rename(columns={"clean_text": "processed"})
        X_tr, _, _, _ = vectorize_splits(train2, val2, test2, text_col="processed")
        self.assertEqual(X_tr.shape[0], len(train2))

    def test_sparse_output(self):
        from scipy.sparse import issparse
        X_tr, X_va, X_te, _ = vectorize_splits(self.train, self.val, self.test)
        self.assertTrue(issparse(X_tr))
        self.assertTrue(issparse(X_va))
        self.assertTrue(issparse(X_te))

    def test_train_only_fits_vectorizer(self):
        # A term only in val/test should NOT appear in the vocabulary
        train3 = self.train.copy()
        val3   = self.val.copy()
        test3  = self.test.copy()
        # Inject a unique term only into val
        val3.loc[val3.index[0], "clean_text"] = "uniquevalidationonlytoken"
        _, _, _, vec = vectorize_splits(train3, val3, test3)
        self.assertNotIn("uniquevalidationonlytoken", vec.vocabulary_)

    def test_accepts_prebuilt_vectorizer(self):
        custom_vec = build_vectorizer(max_features=500)
        X_tr, X_va, X_te, returned_vec = vectorize_splits(
            self.train, self.val, self.test, vectorizer=custom_vec)
        self.assertIs(returned_vec, custom_vec)
        self.assertLessEqual(X_tr.shape[1], 500)


# ---------------------------------------------------------------------------
# Tests: train_lr
# ---------------------------------------------------------------------------

class TestTrainLR(unittest.TestCase):

    def setUp(self):
        train, val, test = make_splits(n_toxic=300, n_clean=700)
        X_tr, X_va, X_te, vec = vectorize_splits(train, val, test)
        self.X_train = X_tr
        self.y_train = train["label"].values
        self.X_val   = X_va
        self.y_val   = val["label"].values

    def test_returns_two_items(self):
        result = train_lr(self.X_train, self.y_train,
                          C_values=[1.0], cv_folds=2, verbose=0)
        self.assertEqual(len(result), 2)

    def test_best_estimator_is_lr(self):
        from sklearn.linear_model import LogisticRegression
        lr, _ = train_lr(self.X_train, self.y_train,
                         C_values=[1.0], cv_folds=2, verbose=0)
        self.assertIsInstance(lr, LogisticRegression)

    def test_grid_search_returned(self):
        from sklearn.model_selection import GridSearchCV
        _, grid = train_lr(self.X_train, self.y_train,
                           C_values=[0.1, 1.0], cv_folds=2, verbose=0)
        self.assertIsInstance(grid, GridSearchCV)

    def test_best_C_in_search_space(self):
        C_values = [0.1, 1.0, 10.0]
        _, grid = train_lr(self.X_train, self.y_train,
                           C_values=C_values, cv_folds=2, verbose=0)
        self.assertIn(grid.best_params_["C"], C_values)

    def test_estimator_is_refitted(self):
        # After GridSearchCV with refit=True, the best estimator can predict
        lr, _ = train_lr(self.X_train, self.y_train,
                         C_values=[1.0], cv_folds=2, verbose=0)
        preds = lr.predict(self.X_val)
        self.assertEqual(len(preds), len(self.y_val))

    def test_class_weight_balanced(self):
        lr, _ = train_lr(self.X_train, self.y_train,
                         C_values=[1.0], cv_folds=2, verbose=0)
        self.assertEqual(lr.class_weight, "balanced")

    def test_predictions_are_binary(self):
        lr, _ = train_lr(self.X_train, self.y_train,
                         C_values=[1.0], cv_folds=2, verbose=0)
        preds = lr.predict(self.X_val)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_proba_sums_to_one(self):
        lr, _ = train_lr(self.X_train, self.y_train,
                         C_values=[1.0], cv_folds=2, verbose=0)
        probs = lr.predict_proba(self.X_val)
        self.assertEqual(probs.shape[1], 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_cv_results_has_expected_keys(self):
        _, grid = train_lr(self.X_train, self.y_train,
                           C_values=[1.0], cv_folds=2, verbose=0)
        self.assertIn("mean_test_score", grid.cv_results_)
        self.assertIn("param_C", grid.cv_results_)

    def test_beats_majority_baseline_on_toxic_f1(self):
        from sklearn.metrics import f1_score
        lr, _ = train_lr(self.X_train, self.y_train,
                         C_values=[0.1, 1.0], cv_folds=2, verbose=0)
        preds = lr.predict(self.X_val)
        f1 = f1_score(self.y_val, preds, pos_label=1, zero_division=0)
        majority_f1 = f1_score(self.y_val, np.zeros(len(self.y_val)),
                               pos_label=1, zero_division=0)
        self.assertGreater(f1, majority_f1,
                           f"LR F1={f1:.4f} should beat majority F1={majority_f1:.4f}")


# ---------------------------------------------------------------------------
# Tests: get_top_features
# ---------------------------------------------------------------------------

class TestGetTopFeatures(unittest.TestCase):

    def setUp(self):
        train, val, test = make_splits(n_toxic=300, n_clean=700)
        X_tr, _, _, self.vec = vectorize_splits(train, val, test)
        self.lr, _ = train_lr(X_tr, train["label"].values,
                              C_values=[1.0], cv_folds=2, verbose=0)

    def test_returns_two_dataframes(self):
        result = get_top_features(self.vec, self.lr, n=10)
        self.assertEqual(len(result), 2)

    def test_toxic_df_has_correct_columns(self):
        toxic_df, _ = get_top_features(self.vec, self.lr, n=10)
        self.assertIn("feature", toxic_df.columns)
        self.assertIn("coefficient", toxic_df.columns)

    def test_correct_number_of_features(self):
        n = 15
        toxic_df, nontoxic_df = get_top_features(self.vec, self.lr, n=n)
        self.assertEqual(len(toxic_df), n)
        self.assertEqual(len(nontoxic_df), n)

    def test_toxic_features_have_positive_coefs(self):
        toxic_df, _ = get_top_features(self.vec, self.lr, n=10)
        self.assertTrue((toxic_df["coefficient"] > 0).all(),
                        "Top toxic features should have positive coefficients")

    def test_nontoxic_features_have_negative_coefs(self):
        _, nontoxic_df = get_top_features(self.vec, self.lr, n=10)
        self.assertTrue((nontoxic_df["coefficient"] < 0).all(),
                        "Top non-toxic features should have negative coefficients")

    def test_features_are_in_vocabulary(self):
        vocab = set(self.vec.get_feature_names_out())
        toxic_df, nontoxic_df = get_top_features(self.vec, self.lr, n=20)
        for feat in toxic_df["feature"]:
            self.assertIn(feat, vocab)
        for feat in nontoxic_df["feature"]:
            self.assertIn(feat, vocab)

    def test_no_overlap_between_top_features(self):
        toxic_df, nontoxic_df = get_top_features(self.vec, self.lr, n=10)
        toxic_set    = set(toxic_df["feature"])
        nontoxic_set = set(nontoxic_df["feature"])
        self.assertEqual(len(toxic_set & nontoxic_set), 0,
                         "The same feature should not be top for both classes")

    def test_model_learns_toxic_keywords(self):
        # Our synthetic toxic texts contain "idiot", "moron", "fool"
        # At least some of these should appear in top toxic features
        toxic_df, _ = get_top_features(self.vec, self.lr, n=30)
        top_words = set(toxic_df["feature"].str.lower())
        toxic_keywords = {"idiot", "moron", "fool", "hate", "stupid", "worthless"}
        found = top_words & toxic_keywords
        self.assertGreater(len(found), 0,
                           f"Expected some toxic keywords in top features, found: {top_words}")


# ---------------------------------------------------------------------------
# Tests: tune_threshold
# ---------------------------------------------------------------------------

class TestTuneThreshold(unittest.TestCase):

    def setUp(self):
        # Simple synthetic probabilities
        np.random.seed(42)
        n = 500
        y_true = np.array([1] * 50 + [0] * 450)   # 9:1 imbalance
        # Generate probs that are somewhat calibrated
        y_prob = np.where(y_true == 1,
                          np.random.beta(5, 2, n),  # toxic: skewed high
                          np.random.beta(2, 5, n))  # clean: skewed low
        self.y_true = y_true
        self.y_prob = y_prob

    def test_returns_two_items(self):
        result = tune_threshold(self.y_true, self.y_prob)
        self.assertEqual(len(result), 2)

    def test_best_threshold_in_range(self):
        best_t, _ = tune_threshold(self.y_true, self.y_prob)
        self.assertGreaterEqual(best_t, 0.05)
        self.assertLessEqual(best_t, 0.95)

    def test_results_df_has_expected_columns(self):
        _, df = tune_threshold(self.y_true, self.y_prob)
        for col in ["threshold", "f1", "precision", "recall"]:
            self.assertIn(col, df.columns)

    def test_results_df_covers_threshold_range(self):
        _, df = tune_threshold(self.y_true, self.y_prob)
        self.assertLessEqual(df["threshold"].min(), 0.1)
        self.assertGreaterEqual(df["threshold"].max(), 0.9)

    def test_metrics_are_bounded(self):
        _, df = tune_threshold(self.y_true, self.y_prob)
        for col in ["f1", "precision", "recall"]:
            self.assertTrue((df[col] >= 0).all())
            self.assertTrue((df[col] <= 1).all())

    def test_custom_threshold_range(self):
        custom_thresholds = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        best_t, df = tune_threshold(self.y_true, self.y_prob,
                                    thresholds=custom_thresholds)
        self.assertEqual(len(df), len(custom_thresholds))
        self.assertIn(best_t, custom_thresholds)

    def test_optimising_recall_gives_lower_threshold(self):
        best_f1,     _ = tune_threshold(self.y_true, self.y_prob, metric="f1")
        best_recall, _ = tune_threshold(self.y_true, self.y_prob, metric="recall")
        # Maximising recall should push threshold lower (catch more positives)
        self.assertLessEqual(best_recall, best_f1 + 0.2,
                             "Recall-optimised threshold should be <= F1-optimised")


# ---------------------------------------------------------------------------
# Tests: save_model / load_model
# ---------------------------------------------------------------------------

class TestSaveLoadModel(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp_dir = Path(tempfile.mkdtemp())
        train, val, test = make_splits(n_toxic=200, n_clean=800)
        X_tr, _, X_te, self.vec = vectorize_splits(train, val, test)
        self.lr, _ = train_lr(X_tr, train["label"].values,
                              C_values=[1.0], cv_folds=2, verbose=0)
        self.X_test    = X_te
        self.test_df   = test

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_creates_pkl_files(self):
        save_model(self.vec, self.lr, models_dir=self.tmp_dir)
        self.assertTrue((self.tmp_dir / "tfidf_vectorizer.pkl").exists())
        self.assertTrue((self.tmp_dir / "logistic_regression.pkl").exists())

    def test_load_returns_two_items(self):
        save_model(self.vec, self.lr, models_dir=self.tmp_dir)
        result = load_model(models_dir=self.tmp_dir)
        self.assertEqual(len(result), 2)

    def test_loaded_vectorizer_produces_same_shape(self):
        save_model(self.vec, self.lr, models_dir=self.tmp_dir)
        vec2, _ = load_model(models_dir=self.tmp_dir)
        X_orig   = self.vec.transform(self.test_df["clean_text"].fillna(""))
        X_reload = vec2.transform(self.test_df["clean_text"].fillna(""))
        self.assertEqual(X_orig.shape, X_reload.shape)

    def test_loaded_model_identical_predictions(self):
        orig_preds = self.lr.predict(self.X_test)
        save_model(self.vec, self.lr, models_dir=self.tmp_dir)
        vec2, lr2 = load_model(models_dir=self.tmp_dir)
        X_reload = vec2.transform(self.test_df["clean_text"].fillna(""))
        reload_preds = lr2.predict(X_reload)
        np.testing.assert_array_equal(orig_preds, reload_preds)

    def test_load_raises_if_files_missing(self):
        with self.assertRaises(FileNotFoundError):
            load_model(models_dir=self.tmp_dir / "nonexistent")


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestNBOWIntegration(unittest.TestCase):

    def test_full_pipeline_runs(self):
        """End-to-end: vectorize → train → threshold → evaluate, no errors."""
        from sklearn.metrics import f1_score

        train, val, test = make_splits(n_toxic=400, n_clean=1600)
        X_tr, X_va, X_te, vec = vectorize_splits(train, val, test)

        lr, grid = train_lr(X_tr, train["label"].values,
                            C_values=[0.1, 1.0], cv_folds=3, verbose=0)

        val_probs = lr.predict_proba(X_va)[:, 1]
        best_t, _ = tune_threshold(val["label"].values, val_probs)

        test_probs = lr.predict_proba(X_te)[:, 1]
        test_preds = (test_probs >= best_t).astype(int)

        f1 = f1_score(test["label"].values, test_preds, pos_label=1, zero_division=0)
        self.assertGreater(f1, 0.0, "Pipeline should produce non-zero F1 on synthetic data")
        self.assertLessEqual(f1, 1.0)

    def test_no_data_leakage_in_vectorization(self):
        """Vocabulary must come from train only."""
        train, val, test = make_splits()

        # Inject a unique token only in the test set
        test = test.copy()
        test.loc[test.index[0], "clean_text"] = "supersecrettesttoken xyz"

        _, _, _, vec = vectorize_splits(train, val, test)
        self.assertNotIn("supersecrettesttoken", vec.vocabulary_)

    def test_top_features_reflect_data(self):
        """The model should learn that toxic keywords predict toxicity."""
        train, val, test = make_splits(n_toxic=500, n_clean=500)
        X_tr, _, _, vec = vectorize_splits(train, val, test)
        lr, _ = train_lr(X_tr, train["label"].values,
                         C_values=[1.0], cv_folds=2, verbose=0)

        toxic_df, nontoxic_df = get_top_features(vec, lr, n=50)
        top_toxic_words = set(toxic_df["feature"].str.lower())

        # At least one of our planted toxic keywords should be in top features
        expected_toxic = {"idiot", "moron", "fool", "hate", "stupid"}
        found = top_toxic_words & expected_toxic
        self.assertGreater(len(found), 0,
                           f"No toxic keywords in top features. Found: {sorted(top_toxic_words)[:10]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestBuildVectorizer,
        TestVectorizeSplits,
        TestTrainLR,
        TestGetTopFeatures,
        TestTuneThreshold,
        TestSaveLoadModel,
        TestNBOWIntegration,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed

    print(f"\n{'='*50}")
    print(f"  {passed}/{total} tests passed {'✓' if failed == 0 else '✗'}")
    print(f"{'='*50}")

    if result.failures:
        print("\nFAILURES:")
        for test, _ in result.failures:
            print(f"  {test}")
    if result.errors:
        print("\nERRORS:")
        for test, _ in result.errors:
            print(f"  {test}")
