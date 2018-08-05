# import plaidml.keras
# plaidml.keras.install_backend()
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import boston_housing
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

max_features = X_train.shape[1]

model = Sequential()
model.add(Dense(32, input_shape=(max_features,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

early_stop = EarlyStopping(
    monitor='val_loss', patience=20, mode='auto', min_delta=0.3, verbose=1)
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=4,
                    epochs=300, callbacks=[early_stop])

model.summary()

Y_pred = model.predict(X_test)

print("Y_test", Y_test[:5])
print("Y_pred", Y_pred[:5, 0])

# Plot the Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Plot the Accuracy Curves
# plt.figure(figsize=[8, 6])
# plt.plot(history.history['mean_squared_error:'], 'r', linewidth=3.0)
# plt.plot(history.history['val_mean_squared_error:'], 'b', linewidth=3.0)
# plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
# plt.xlabel('Epochs ', fontsize=16)
# plt.ylabel('Accuracy', fontsize=16)
# plt.title('Accuracy Curves', fontsize=16)

plt.show()
