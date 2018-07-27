from mnistdnn import MnistDNN

MnistDNN.get_data()

dnn = MnistDNN(
    hidden_layers=(500, 100, 30),
    learning_rate=1e-3)

dnn.build()
dnn.run()
