"""Step 2 — application scorecard (WoE + logistic regression).

Classic, explainable credit-scoring approach:
- bin every feature on the development sample and replace values with Weight of
  Evidence (WoE); report Information Value (IV) per feature;
- fit a logistic regression on the WoE features;
- scale log-odds to points (base 600 at 4:1 good:bad odds, 40 points to double);
- validate out-of-time (cohorts before Jun-2025 = development, later = OOT) with
  AUC / Gini / KS and a Population Stability Index (PSI).

`int_rate` is deliberately excluded: it is LendingClub's own risk-based price,
so using it would just borrow their scorecard. `addr_state` is excluded on
fairness and stability grounds.
"""
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

import config as C
from plotting import style, save

NUMERIC = ["loan_amnt", "annual_inc", "dti", "revol_util", "total_acc",
           "longest_credit_length", "emp_length", "delinq_2yrs"]
CATEGORICAL = ["term", "home_ownership", "purpose", "verification_status",
               "emp_length_missing"]
TARGET = "bad_loan"
OOT_FROM_VINTAGE = C.OOT_FROM_VINTAGE


def fit_bins(s: pd.Series, n_bins: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def woe_table(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"x": x, "y": y}).groupby("x", observed=True)["y"].agg(["count", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    # +0.5 smoothing avoids infinite WoE in sparse bins
    dist_g = (t["good"] + 0.5) / (t["good"].sum() + 0.5 * len(t))
    dist_b = (t["bad"] + 0.5) / (t["bad"].sum() + 0.5 * len(t))
    t["bad_rate"] = t["bad"] / t["n"]
    t["woe"] = np.log(dist_g / dist_b)
    t["iv"] = (dist_g - dist_b) * t["woe"]
    return t


class WoEEncoder:
    def fit(self, df, y):
        self.edges, self.tables = {}, {}
        for col in NUMERIC + CATEGORICAL:
            x = self._bin(df[col], col, fit=True)
            self.tables[col] = woe_table(x, y)
        return self

    def _bin(self, s, col, fit=False):
        if col in NUMERIC:
            if fit:
                self.edges[col] = fit_bins(s)
            return pd.cut(s, self.edges[col]).astype(str)
        return s.astype(str)

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for col, t in self.tables.items():
            out[col] = self._bin(df[col], col).map(t["woe"]).fillna(0.0)
        return out

    def iv(self):
        return pd.Series({c: t["iv"].sum() for c, t in self.tables.items()}).sort_values(ascending=False)


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(expected, actual, edges):
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def to_score(pd_):
    factor = C.SCORE_PDO / np.log(2)
    offset = C.SCORE_BASE - factor * np.log(C.SCORE_BASE_ODDS)
    return np.round(offset + factor * np.log((1 - pd_) / pd_)).astype(int)


def main():
    style()
    con = duckdb.connect(str(C.DB_PATH))
    df = con.sql("SELECT * FROM loans").df()
    dev = df[df.vintage_idx < OOT_FROM_VINTAGE]
    oot = df[df.vintage_idx >= OOT_FROM_VINTAGE]

    enc = WoEEncoder().fit(dev, dev[TARGET])
    iv = enc.iv()
    keep = iv[iv >= 0.02].index.tolist()          # drop "useless" predictors
    X_dev, X_oot = enc.transform(dev)[keep], enc.transform(oot)[keep]

    model = LogisticRegression(C=1.0, max_iter=1000).fit(X_dev, dev[TARGET])
    df["pd"] = model.predict_proba(enc.transform(df)[keep])[:, 1]
    df["score"] = to_score(df["pd"])
    df["sample"] = np.where(df.vintage_idx < OOT_FROM_VINTAGE, "dev", "oot")

    # risk bands = score quintiles on dev (A best)
    cuts = np.quantile(df.loc[df["sample"] == "dev", "score"], [0.2, 0.4, 0.6, 0.8])
    df["band"] = pd.cut(df["score"], [-np.inf, *cuts, np.inf], labels=list("EDCBA")).astype(str)

    metrics = []
    for name, part in df.groupby("sample"):
        auc = roc_auc_score(part[TARGET], part["pd"])
        metrics.append({"sample": name, "n": len(part), "bad_rate": part[TARGET].mean(),
                        "auc": auc, "gini": 2 * auc - 1, "ks": ks_stat(part[TARGET], part["pd"])})
    metrics = pd.DataFrame(metrics)
    edges = np.quantile(df.loc[df["sample"] == "dev", "score"], np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    metrics["psi_vs_dev"] = [0.0, psi(df.loc[df["sample"] == "dev", "score"], df.loc[df["sample"] == "oot", "score"], edges)]

    coefs = pd.DataFrame({"feature": keep, "coef": model.coef_[0], "iv": iv[keep].values})
    band_tbl = (df.groupby(["band", "sample"])
                .agg(n=("loan_id", "size"), avg_pd=("pd", "mean"), bad_rate=(TARGET, "mean"),
                     min_score=("score", "min"), max_score=("score", "max"))
                .reset_index().sort_values(["sample", "band"]))

    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(C.TAB_DIR / "scorecard_metrics.csv", index=False)
    iv.round(4).rename("iv").to_csv(C.TAB_DIR / "information_value.csv")
    coefs.round(4).to_csv(C.TAB_DIR / "scorecard_coefficients.csv", index=False)
    band_tbl.round(4).to_csv(C.TAB_DIR / "risk_band_calibration.csv", index=False)

    con.register("scores_df", df[["loan_id", "pd", "score", "band", "sample"]])
    con.execute("CREATE OR REPLACE TABLE scores AS SELECT * FROM scores_df")
    con.close()

    # --- figures --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, part in df.groupby("sample"):
        fpr, tpr, _ = roc_curve(part[TARGET], part["pd"])
        g = metrics.set_index("sample").loc[name, "gini"]
        axes[0].plot(fpr, tpr, label=f"{name.upper()} (Gini {g:.2f})")
    axes[0].plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
    axes[0].set(title="ROC — development vs out-of-time", xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend()
    b = band_tbl.pivot(index="band", columns="sample", values="bad_rate").loc[list("ABCDE")]
    b.plot.bar(ax=axes[1], rot=0)
    axes[1].set(title="Observed bad rate by risk band", xlabel="Band (A = lowest risk)", ylabel="Bad rate")
    axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    save(fig, "02_scorecard_performance")
    print(metrics.round(3).to_string(index=False))
    print(iv.round(3).to_string())


if __name__ == "__main__":
    main()
