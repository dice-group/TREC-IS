from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize


class Model:
    def __init__(self, X, y):
        '''
        :param X: training features
        :param y: categories
        '''
        # Scale and normalize training features
        self.X = scale(X)
        self.X = normalize(X, norm='l2')

        # encode y classes to one-hot
        self.y_encoded = to_categorical(y)

        self.num_classes = 23  # number of classes (information types)
        self.features_len = 301

        self.X_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y_encoded,
                                                                                test_size=0.1)

    def simple_DeepModel(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=self.features_len))  # 301= 300 (embedding) + 1 (sentiment)
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.X_train, self.y_train, epochs=30, batch_size=32, verbose=2, shuffle=True)

        return model

    def evaluate_model(self):
        model = self.simple_DeepModel()
        score = model.evaluate(self.x_test, self.y_test, batch_size=32, verbose=2)

        print(score[1])
