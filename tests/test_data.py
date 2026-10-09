"""Covers synopsis TC1: preprocessing produces no nulls, correct encoding,
and the generated customer_value feature."""


def test_preprocess_has_no_nulls(processed_df):
    assert processed_df.isnull().sum().sum() == 0


def test_preprocess_all_columns_numeric(processed_df):
    non_numeric = processed_df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    assert non_numeric == []


def test_customer_value_feature_present_and_positive(processed_df):
    assert "customer_value" in processed_df.columns
    assert (processed_df["customer_value"] >= 0).all()


def test_customer_value_is_monthly_revenue_not_lifetime(processed_df):
    """Paper Eq. 3: CLV uses the customer's MONTHLY revenue R_i - the
    1/(1 - r) denominator already capitalises future periods, so
    customer_value must NOT be scaled by tenure (the old
    MonthlyCharges * (tenure + 1) double-counted tenure and inflated every
    CLV/e-Profits figure)."""
    assert (processed_df["customer_value"] == processed_df["MonthlyCharges"]).all()


def test_customerid_dropped(processed_df):
    assert "customerID" not in processed_df.columns


def test_split_is_stratified_and_disjoint(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    assert set(x_train.index).isdisjoint(set(x_test.index))
    # churn rate shouldn't drift wildly between splits
    assert abs(y_train.mean() - y_test.mean()) < 0.15


def test_balance_classes_smote_equalizes_training_classes(small_split):
    from pipeline import data

    (x_train, _, y_train, _), _ = small_split
    x_bal, y_bal = data.balance_classes(x_train, y_train, method="smote", random_state=0)
    counts = y_bal.value_counts()
    assert counts.min() == counts.max()


def test_balance_classes_none_is_noop(small_split):
    from pipeline import data

    (x_train, _, y_train, _), _ = small_split
    x_bal, y_bal = data.balance_classes(x_train, y_train, method=None)
    assert x_bal is x_train
    assert y_bal is y_train
