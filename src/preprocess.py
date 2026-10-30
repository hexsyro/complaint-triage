
import pandas as pd
import joblib


from scipy.sparse import save_npz

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def split(df):
    X_train, X_test, y_train, y_test = train_test_split(
    df['Issue'], df['Product'], test_size=0.2,
    random_state=42, stratify=df['Product'])


    tartget_comp=pd.DataFrame({'train': y_train.value_counts(normalize=True),
                'test': y_test.value_counts(normalize=True)})
    
    return tartget_comp,X_train, X_test, y_train, y_test

def vectorize(X_train, X_test):
    vec = TfidfVectorizer(stop_words='english')
    train_matrix = vec.fit_transform(X_train)
    #converts the test texts using what vec already learned.
    test_matrix = vec.transform(X_test)

    return train_matrix, test_matrix, vec

def export(train_matrix, test_matrix, y_train, y_test, X_test):
    save_npz('data/processed/x_train.npz', train_matrix)
    save_npz('data/processed/x_test.npz', test_matrix)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    X_test.to_csv('data/processed/x_text.csv', index=False)

def main():
    df = pd.read_csv('data/processed/complaints_sample.csv')

    tartget_comp,X_train, X_test, y_train, y_test=split(df)
    print()
    print()
    print(tartget_comp)
    print()

    train_matrix,test_matrix,vec=vectorize(X_train, X_test)
    print('text vectorization completed successfully!')
    print()

    joblib.dump(vec, 'models/vectorizer_model.joblib')
    print("model file exporting completed successfully!")

    export(train_matrix, test_matrix, y_train, y_test, X_test)
    print("input and target files successfully exported into 'data/processed' folder...")


if __name__ == '__main__':
    main()