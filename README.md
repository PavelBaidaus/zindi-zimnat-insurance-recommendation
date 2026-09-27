# Insurance product recommendation (Zindi × Zimnat, 2020)

**Top 10% of the final leaderboard, solo.** Competition: Zimnat Insurance Recommendation Challenge on Zindi.

## Task

Each Zimnat customer holds several of 21 insurance products. In the test set one product per customer is hidden, and the task is to predict which one it is. Metric: log loss.

## Approach

- **Reformulation:** every product a training customer holds becomes one example with that product hidden, which turns the task into 21-class classification.
- **Features:** demographics, join-date parts (year, quarter, month, day), product-ownership flags, and categorical encodings.
- **Model:** soft-voting ensemble of LightGBM, CatBoost (GPU) and XGBoost (mlxtend `EnsembleVoteClassifier`).
- **Calibration:** ensemble probabilities are recalibrated with splines (`ml-insights` `SplineCalib`) before submission.

Best public-leaderboard log loss: 0.0267.

## Files

| File | Contents |
|---|---|
| `Zindiinsurance(Tikhon startpack)-LGBM-Best0.0266636821396398public-votestackspline.ipynb` | Full pipeline: reshaping, features, ensemble, calibration, submission |

## Credits

Builds on a public starter notebook by Tikhon for this competition.

**Stack:** Python, pandas, LightGBM, CatBoost, XGBoost, mlxtend, ml-insights.
