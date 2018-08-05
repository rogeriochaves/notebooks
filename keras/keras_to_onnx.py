import onnxmltools
from keras.layers import Input, Dense, Add
from keras.models import Model


# N: batch size, C: sub-model input dimension, D: final model's input dimension
N, C, D = 2, 3, 3

# Define a sub-model, it will become a part of our final model
sub_input1 = Input(shape=(C,))
sub_mapped1 = Dense(D)(sub_input1)
sub_model1 = Model(inputs=sub_input1, outputs=sub_mapped1)

# Define another sub-model, it will become a part of our final model
sub_input2 = Input(shape=(C,))
sub_mapped2 = Dense(D)(sub_input2)
sub_model2 = Model(inputs=sub_input2, outputs=sub_mapped2)

# Define a model built upon the previous two sub-models
input1 = Input(shape=(D,))
input2 = Input(shape=(D,))
mapped1_2 = sub_model1(input1)
mapped2_2 = sub_model2(input2)
sub_sum = Add()([mapped1_2, mapped2_2])
keras_model = Model(inputs=[input1, input2], outputs=sub_sum)

# Convert it!
onnx_model = onnxmltools.convert_keras(keras_model)
onnxmltools.utils.save_text(onnx_model, 'onnx_example.json')
onnxmltools.utils.save_model(onnx_model, 'onnx_example.onnx')
