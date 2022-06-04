import os
import random
import string

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# def substitute(df):
#     df['Sex'] = [1 if i == 'male' else 0 for i in df['Sex']]

#     # Embarked: S = 2 C = 1 Q = 0, NaN = 5
#     temp = df['Embarked']
#     temp_list = []
#     for i in temp:
#         if i == 'S':
#             temp_list.append(2)
#         elif i == 'C':
#             temp_list.append(1)
#         elif i == 'Q':
#             temp_list.append(0)
#         else:
#             temp_list.append(2)
#     df['Embarked'] = temp_list

#     # if Age == NaN -> Age = Age.mean()
#     df['Age'] = df['Age'].fillna(df['Age'].mean())

#     # if Fare == NaN -> Fare = Fare.mean()
#     df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    
#     return df

# def preprocess(df, test_df=None):
#     df = substitute(df)

#     scaler = MinMaxScaler()
#     ret = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Perished']]
#     # Scaling to 0~1.
#     scaler.fit(ret)
#     ret = scaler.transform(ret)

#     if test_df is not None:
#         test_df = substitute(test_df)

#         scaler = MinMaxScaler()
#         scaler.fit(df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']])

#         test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']] = \
#             scaler.transform(test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']])

#     return ret, test_df


def fix_seed(seed=1234):
    """ シードの固定
    Args:
        seed (int): A random seed. default to 1234.
    """    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str('1234')
    np.random.seed(seed)
    torch.manual_seed(seed)

def devide_df(df, length_train):
    return df.loc[:length_train - 1], df.loc[length_train:].drop(['Survived'], axis=1)

def preprocess(train_path, test_path):
    def extract_surname(data):    
    
        families = []
        
        for i in range(len(data)):        
            name = data.iloc[i]

            if '(' in name:
                name_no_bracket = name.split('(')[0] 
            else:
                name_no_bracket = name
                
            family = name_no_bracket.split(',')[0]
            title = name_no_bracket.split(',')[1].strip().split(' ')[0]
            
            for c in string.punctuation:
                family = family.replace(c, '').strip()
                
            families.append(family)
                
        return families

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_all = pd.concat([df_train, df_test], sort=True)
    # インデックス番号の振り直し
    df_all = df_all.reset_index()

    # 年齢の欠損値を性別と客室等級ごとの平均で埋める
    df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

    # 年齢をある程度の範囲でグルーピング
    df_all['Age'] = pd.qcut(df_all['Age'], 10)

    # 乗船港の欠損値を最頻値で埋める
    df_all['Embarked'] = df_all['Embarked'].fillna(df_all['Embarked'].mode().iloc[0])

    # チケット料金を乗船していた家族の数とチケットのクラスごとの中央値で埋める
    # 乗船していた家族の人数をdfに追加
    df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1

    # Family_Sizeをある程度まとめる
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
    df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)

    # チケット料金の欠損値を客室クラスと家族の人数の中央値で埋める
    df_all['Fare'] = df_all.groupby(['Pclass', 'Family_Size'])['Fare'].apply(lambda x: x.fillna(x.median()))
    df_all['Fare'] = pd.qcut(df_all['Fare'], 13)    

    # キャビンの位置を分類：欠損値はMとしてそのまま利用
    df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    # 似た属性の位置をひとまとめに
    df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C', 'T'], 'ABCT')
    df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
    df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
    # 元のキャビンのカラムを消す
    df_all = df_all.drop(['Cabin'], axis=1)

    # 称号(Mr., Ms.的な敬称的なやつ)を分類
    df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    
    # 結婚している女性は生存率が高いので特徴量として追加
    df_all['Is_Married'] = 0
    df_all.loc[df_all['Title'] == 'Mrs', 'Is_Married'] = 1

    # 似た属性の敬称をまとめる
    # 女性
    df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
    # 称号的なやつ
    df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

    # 姓を取得
    df_all['Family'] = extract_surname(df_all['Name'])

    # 同じチケットの人を取得
    df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')

    # 生存率を出すためにtest, trainを分離(trainのほうしか生存率を出せない)
    df_train = df_all.loc[:890]
    df_test = df_all.loc[891:]

    # trainとtestにまたがって，同じ家族や，同じチケットの人がいるので，それをリストアップ
    non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
    non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]
    
    df_family_perished_rate = df_train.groupby('Family')[['Perished', 'Family_Size']].median()
    df_ticket_perished_rate = df_train.groupby('Ticket')[['Perished', 'Ticket_Frequency']].median()

    family_rates = {}
    ticket_rates = {}

    # 家族ごとの死亡率を取得
    for i in range(len(df_family_perished_rate)):
        # 姓がテストデータにあって，学習データ内でFamily_Sizeが2人以上なら
        if df_family_perished_rate.index[i] in non_unique_families and df_family_perished_rate.iloc[i, 1] > 1:
            # その家族の平均死亡率を取得
            family_rates[df_family_perished_rate.index[i]] = df_family_perished_rate.iloc[i, 0]

    # 同じチケットの人の死亡率を取得
    for i in range(len(df_ticket_perished_rate)):
        # 同じチケットNo.の人がテストデータにもあって，そのチケットナンバーが学習データ内で2回以上出現するものなら
        if df_ticket_perished_rate.index[i] in non_unique_tickets and df_ticket_perished_rate.iloc[i, 1] > 1:
            # そのチケットNo.の人の平均死亡率を取得
            ticket_rates[df_ticket_perished_rate.index[i]] = df_ticket_perished_rate.iloc[i, 0]
    
    mean_perished_rate = np.mean(df_train['Perished'])

    train_family_perished_rate = []
    train_family_perished_rate_NA = []
    test_family_perished_rate = []
    test_family_perished_rate_NA = []

    for data in df_train['Family'].values:
        # trainデータに該当する姓の死亡率の情報があれば，その死亡率を追加
        if data in family_rates:
            train_family_perished_rate.append(family_rates[data])
            train_family_perished_rate_NA.append(1)
        # なければ平均死亡率を追加
        else:
            train_family_perished_rate.append(mean_perished_rate)
            train_family_perished_rate_NA.append(0)

    for data in df_test['Family'].values:
        # trainデータに該当する姓の死亡率の情報があれば，その死亡率を追加
        if data in family_rates:
            test_family_perished_rate.append(family_rates[data])
            test_family_perished_rate_NA.append(1)
        # なければ平均死亡率を追加
        else:
            test_family_perished_rate.append(mean_perished_rate)
            test_family_perished_rate_NA.append(0)


    train_ticket_perished_rate = []
    train_ticket_perished_rate_NA = []
    test_ticket_perished_rate = []
    test_ticket_perished_rate_NA = []

    for data in df_train['Ticket'].values:
        # trainデータに該当する姓の死亡率の情報があれば，その死亡率を追加
        if data in ticket_rates:
            train_ticket_perished_rate.append(ticket_rates[data])
            train_ticket_perished_rate_NA.append(1)
        # なければ平均死亡率を追加
        else:
            train_ticket_perished_rate.append(mean_perished_rate)
            train_ticket_perished_rate_NA.append(0)

    for data in df_test['Ticket'].values:
        # trainデータに該当する姓の死亡率の情報があれば，その死亡率を追加
        if data in ticket_rates:
            test_ticket_perished_rate.append(ticket_rates[data])
            test_ticket_perished_rate_NA.append(1)
        # なければ平均死亡率を追加
        else:
            test_ticket_perished_rate.append(mean_perished_rate)
            test_ticket_perished_rate_NA.append(0)

    df_train = pd.concat(
        [
            df_train,
            pd.DataFrame(train_family_perished_rate, columns=['Family_Perished_Rate']),
            pd.DataFrame(train_family_perished_rate_NA, columns=['Family_Perished_Rate_NA']),
            pd.DataFrame(train_ticket_perished_rate, columns=['Ticket_Perished_Rate']),
            pd.DataFrame(train_ticket_perished_rate_NA, columns=['Ticket_Perished_Rate_NA']),
        ], axis=1)

    df_test = df_test.reset_index()
    df_test = pd.concat(
        [
            df_test,
            pd.DataFrame(test_family_perished_rate, columns=['Family_Perished_Rate']),
            pd.DataFrame(test_family_perished_rate_NA, columns=['Family_Perished_Rate_NA']),
            pd.DataFrame(test_ticket_perished_rate, columns=['Ticket_Perished_Rate']),
            pd.DataFrame(test_ticket_perished_rate_NA, columns=['Ticket_Perished_Rate_NA']),
        ], axis=1)

    for df in [df_train, df_test]:
        df['Perished_Rate'] = (df['Ticket_Perished_Rate'] + df['Family_Perished_Rate']) / 2
        df['Perished_Rate_NA'] = (df['Ticket_Perished_Rate_NA'] + df['Family_Perished_Rate_NA']) / 2

    # 非数値特徴量を数値に変換
    non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

    # sklearnのラベルエンコーダーで数値割り当て
    for df in [df_train, df_test]:
        for feature in non_numeric_features:
            df[feature] = LabelEncoder().fit_transform(df[feature])

    # 複数のクラスがある特徴量(数値事態に意味を持たないもの：年齢やチケット料金は数値の大小に意味があるのでそのまま)をOne-Hotベクトルに変換
    categorical_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']
    encoded_features = []

    for df in [df_train, df_test]:
        for feature in categorical_features:
            encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
            n = df[feature].nunique()
            cols = [f'{feature}_{n}' for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)

    df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
    df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)

    drop_columns = [
        'index',
        'Name',
        'Ticket',
        'Parch',
        'SibSp',
        'PassengerId',
        'Family_Size',
        'Family',
        'Embarked',
        'Sex',
        'Deck',
        'Title',
        'Family_Size_Grouped',
        'Pclass',
        'Ticket_Perished_Rate',
        'Family_Perished_Rate',
        'Ticket_Perished_Rate_NA',
        'Family_Perished_Rate_NA'
    ]
    
    df_train = df_train.drop(drop_columns, axis=1)
    df_test = df_test.drop([*drop_columns, 'Perished', 'level_0'], axis=1)

    # カラムの順番を変更(生死を最後に持ってくる)
    df_train_tmp = df_train.drop(['Perished'], axis=1)
    df_train = pd.concat([df_train_tmp, df_train['Perished']], axis=1)

    return df_train.values, df_test.values

def regularization(data):
    return StandardScaler().fit_transform(data)
