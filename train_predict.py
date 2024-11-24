from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.utils.constants import SEED as DEFAULT_SEED
import pandas as pd
from model import MyModel
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # Only show error messages


def get_prediction(test_path, train_path) -> pd.DataFrame:
    # Read data
    test = pd.read_parquet(test_path)
    train = pd.read_parquet(train_path)

    # Filtering data about 1000 of the most active and all test users
    user_interactions = train.groupby('user_id').size().reset_index(name='counts')
    top_users = user_interactions.sort_values('counts', ascending=False).head(1000)['user_id']
    test_users = test['user_id'].unique()
    combined_users = pd.concat([pd.Series(top_users), pd.Series(test_users)]).unique()
    df = train[train['user_id'].isin(combined_users)]
    df.rename(columns={'user_id': 'userID', 'item_id': 'itemID'}, inplace=True)

    # Choose the most attractive items for every user
    df = df.groupby('userID').apply(lambda x: x.nlargest(5, 'rating')).reset_index(drop=True)

    # Split data
    train_data, test_data = python_stratified_split(df, ratio=0.75)
    data = ImplicitCF(train=train_data, test=test_data, seed=DEFAULT_SEED)

    # Model
    model = MyModel('./lightgcn.yaml')
    model.fit(data)

    # Predict
    test = test.rename(columns={'user_id': 'userID', 'item_id': 'itemID'})
    result = model.predict(test)

    # Convert to output format
    grouped = result.groupby('userID').apply(lambda x: x.sort_values('prediction', ascending=False))
    grouped = grouped.reset_index(drop=True)
    result_df = grouped.groupby('userID')['itemID'].agg(lambda x: list(x)).reset_index()
    result_df.rename(columns={'itemID': 'item_ids'}, inplace=True)

    return result_df


def main():
    path = './data/'  # Path to data folder

    pred_smm = get_prediction(path + 'test_smm.parquet', path + 'train_smm.parquet')
    pred_zvuk = get_prediction(path + 'test_zvuk.parquet', path + 'train_zvuk.parquet')

    pred_smm.to_parquet('submission_smm.parquet')
    pred_zvuk.to_parquet('submission_zvuk.parquet')


if __name__ == "__main__":
    main()
