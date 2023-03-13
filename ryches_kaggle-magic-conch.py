from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format("../input/kaggle-tuned-word2vec/kaggleword2vec.bin", binary = True)
w2v.most_similar("unet")
w2v.most_similar("augmentation")
w2v.most_similar("image")
w2v.most_similar("segmentation")
w2v.most_similar("pneumothorax")
w2v.most_similar("xray")
w2v.most_similar("cnn")
w2v.most_similar(positive=['woman', 'king'], negative=['man'])
w2v.most_similar(positive=['cnn', 'rnn'], negative=['convolution'])