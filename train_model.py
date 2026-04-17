import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/train.csv')

# Fix FutureWarning — modern pandas syntax
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S')

# Feature Engineering
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
)
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
)

# Fix NaN before converting to int
df['FareBin'] = pd.cut(
    df['Fare'],
    bins=[0, 7.9, 14.4, 31.0, 512],
    labels=[0, 1, 2, 3]
)
df['FareBin'] = df['FareBin'].cat.add_categories(-1).fillna(-1).astype(int)

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked', 'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareBin']

df = df[features + ['Survived']].dropna()
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'AgeGroup'], drop_first=False)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"✅ Accuracy: {model.score(X_test, y_test):.3f}")

# Save with current Python version protocol
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ titanic_model.pkl saved")
print("✅ feature_columns.pkl saved")
print(f"📋 Feature count: {len(X.columns)}")
