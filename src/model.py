import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        # adding batchnorm
        # since we are dealing with embedded layer with is 1-d ( 1xembed_size), 1d BatchNorm would be required
        self.batchnorm = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        # adding batch normalization
        features = self.batchnorm(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size =  embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        # lstm 
        self.lstm = nn.LSTM(
            self.embed_size, 
            self.hidden_size,
            self.num_layers,
            dropout=0.5,
            batch_first=True
        )
        
        # dropout this might not be used
        self.dropout = nn.Dropout(0.5)
        
        # liner
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        
    def forward(self, features, captions):
        
        # prep captions
        captions = captions[:, :-1] #<-- controlling for dimensions by removing <end>
        captions = self.embed(captions)
        
        # prep features
        features = features.unsqueeze(1) #<-- converting to timestep, adding caption-length dim as 2nd dimension
        
        # concatenating features and captions into tesnor
        lstm_inputs = torch.cat((features, captions), dim=1)
        print(f"shape of the lstm inputs is: {lstm_inputs.shape}\n")

        # output of the lstm
        output, _ = self.lstm(lstm_inputs)
        
        # fc 
        output = self.fc(output)
        
        return output
        
        
        
    def sample(self, inputs, states=None, max_len=20):
        '''accepts pre-processed image tensor (inputs) and returns 
        predicted sentence (list of tensor ids of length max_len) '''
        
        
        output_sentence = [] #<-- sentence output
        
        for i in range(max_len):

            # predicting the word in the caption
            output, states = self.lstm(inputs, states)
            output = self.fc(output)
            prediction_idx = torch.argmax(output, dim=2) #<-- gettting the idx of highest P word
            
            # build the sentence
            output_sentence.append(prediction_idx.item()) 
            
            # prevent caption going past <end>
            if prediction_idx == 1:
                break
            
            # embed the inputs for next word
            inputs = self.embed(prediction_idx)
                
        return output_sentence
        
        
        
        