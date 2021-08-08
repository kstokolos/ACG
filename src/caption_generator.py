import os
from typing import Callable
from .model import EncoderCNN, DecoderRNN
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
from torchvision import transforms
import numpy as np
from PIL import Image
print('Imports Succesfull')


VOCABULARY_FILE = "vocab.pickle"
ENCODER_FILE = "encoder-3.pkl"
DECODER_FILE = "decoder-3.pkl"
EMBED_SIZE = 256
HIDDEN_SIZE = 512
VOCAB_SIZE = 9955


def setup_ml():
    """
    Set up all the imports and objects required for captions generation
    """
    # load vocabulary
    vocab = open(f"{VOCABULARY_FILE}", "rb")
    vocab = pickle.load(vocab)

    # transformer to preprocess images
    transform_test = transforms.Compose([ 
        transforms.Resize(256),                         
        transforms.RandomCrop(224),                      
        transforms.RandomHorizontalFlip(),               
        transforms.ToTensor(),                           
        transforms.Normalize((0.485, 0.456, 0.406),      
                            (0.229, 0.224, 0.225))])

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(EMBED_SIZE)
    encoder.eval()
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE)
    decoder.eval()

    # load encoder
    encoder.load_state_dict(
        torch.load(
            os.path.join('./models', ENCODER_FILE),
            map_location=torch.device('cpu')
        )
    )

    # load decoder
    decoder.load_state_dict(
        torch.load(
            os.path.join('./models', DECODER_FILE),
            map_location=torch.device('cpu')
        )
    )
    print("\n-- Model components were imported succesfully! -- \n")
    return transform_test, encoder, decoder, vocab


def generate_captions(image_path: str,
                      vocab: dict,
                      transformer: Callable[[], torch.tensor],
                      encoder: Callable[[torch.tensor], torch.tensor],
                      decoder: Callable[[torch.tensor], "list[int]"]) -> str:
    """
    Generates Captions
    :image_path: string, path to the uploaded image
    :vocab: dict, dictionary of indexed vocabulary 
    :return: string
    """

    # Load image
    PIL_image = Image.open(image_path).convert('RGB')
    orig_image = np.array(PIL_image)
    image = transformer(PIL_image)

    # add batch size
    image = image.unsqueeze(0)

    # get features
    features = encoder(image).unsqueeze(1)

    # Pass the embedded image features through the model to get a predicted caption.
    output = decoder.sample(features)
    assert (type(output)==list), "Output needs to be a Python list" 
    assert all([type(x)==int for x in output]), "Output should be a list of integers." 

    # return captions
    return ' '.join([vocab[w] for w in output if w not in [0, 1]])

if __name__ == "__main__":
    transformer, encoder, decoder, vocab = setup_ml()
    captions = generate_captions('./static/coco_img2.png', vocab, transformer, encoder, decoder)
    # print(captions)