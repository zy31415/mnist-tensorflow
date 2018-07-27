from mnistdnn import MnistDNN

MnistDNN.get_data()

dnn = MnistDNN(
    hidden_layers=(100, 30),
    learning_rate=0.1)

dnn.build()
dnn.run()
