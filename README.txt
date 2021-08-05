-- UI/ UX

    H2O.ai Wave App framework is used for front end and user interaction (GIU). 
    https://www.h2o.ai/products/h2o-wave/
    A user simply has to upload and image, that he / she would like to have captions
    generated for. Scoring is done on the backend and doesn't involve any manual 
    code tweaking / running. 


-- Training Dataset

    COCO dataset (Common Objects in Context) is data that the model is trained on. This dataset
    maps images to captions describing what is on those images. Each image has about 5 captions
    on average. https://cocodataset.org/#home


-- CNN

    Convolutional neural networks are the advanced image processing artificial neural networks used in 
    image processing. Main idea is that the image is taken through multiple layers of the network,
    thorough which feature set is exatracted. At the end layer feature object is flattened and can 
    be fed into final classification layer or be combined with other layers.


-- RNN

    Recurrent Neural Networks allow for introduction of memory to neural network architectures and are 
    very helpful in analyzing sequential data like text or time series. However, sometimes we use RNN
    to analyze sequence of images, in other words, frames in a video. Problems like gesture captioning
    spatial patterns, video classification, etc.


-- LSTM

    LSTM or Long Short Term Memory networks are RNNs architecture which allows to overcome a vanishing
    gradient problem. Since, the vanishing gradient is no longer of a major concern, such architecture
    allows for longer sequence modeling.


-- Encoder/Decoder

    Our model will take an image and process it using CNN, then it will take output of the CNN and 
    use it as input to RNN to generate the automatic captions. That's it;)
    Such architecture is know as encoder - decoder network. This means that we take an input, in 
    this case an image, and encode it into features. Then we take those features and decode it into 
    text. 

    Encoder:
    Note, that a special type of CNN will be used, which is called ResNet ( Residual Network ).
    Think about this part as a feature extractor that compresses image into smaller feature representation.

    Connector:
    In order to connect encoder to decoder we get rid of final layer ( Softmax ) in the ResNet and and
    additional Linear Fully Connecter layer to act as the first sequence to the input of the decoder. Once
    feautres extracted and processed to fully connected layer, what is left for decoder is to take the 
    feature vector and decode it into natural language.

    Decoder:
    We use LSTM network as decoder. Before feeding data into LSTM we add Embedding Layer to transform features
    into vectors of the same shape. Input to the decoder network will be words of the captions.


-- Libraries and Tools used:
    
    NLTK
    PyTorch


