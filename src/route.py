import joblib
from scipy.special import softmax


def route_complaint(text, m, vec):
    Y = vec.transform([text.lower().strip()])
    proba = m.predict_proba(Y)[0]
    best=proba.argmax()
    return m.classes_[best], proba[best]


def main():
    m=joblib.load('models/LinearSVC_model.joblib') # wrapped around CalibratedClassifierCV()
    vec=joblib.load('models/vectorizer_model.joblib')
    print()
    print()

    while True:
        text=input("write a complaint to predict the product: ")
        if text == 'q':
            break
        else:
            output, proba = route_complaint(text, m, vec)

            print()
            print("=== Here is the Prediction report ===")
            print()
            print(f"the product is likely a: {output}")
            print(f"the confidence of the prediction is: {proba:.2%}")
            print()
            print("type 'q' to quit...")


if __name__ == '__main__':
    main()


