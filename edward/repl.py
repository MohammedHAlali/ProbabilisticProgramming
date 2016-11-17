import edward as ed

hidden1 = Dense(15, activation=K.relu)(X)

total_parameters = 0
print()
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    print( variable.name , shape , variable_parametes )
    total_parameters += variable_parametes
print(total_parameters)

15 + 15 + 15*15 + 15 + (300 + 20)*3


