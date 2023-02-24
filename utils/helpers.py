import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_feature_importances(model, title, features):

    feature_importances_df = pd.DataFrame(
        {
            "Feature Importance": model.feature_importances_,
            "Feature Name": [v for v in features],
        }
    )

    plt.figure(figsize=(20, 13))

    sns.barplot(
        x="Feature Importance",
        y="Feature Name",
        data=feature_importances_df.loc[
            feature_importances_df["Feature Importance"] > 0
        ].sort_values(by="Feature Importance", ascending=False),
    )

    plt.title("Feature Importances - {}".format(title))
    plt.show()
