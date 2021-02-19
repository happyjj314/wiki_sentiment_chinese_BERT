from torch import nn
from models.bert_model import *


# CLS对应的一条向量


class Bert_Sentiment_Analysis(nn.Module):
    def __init__(self,config):
        super(Bert_Sentiment_Analysis, self).__init__()
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size,1)
        self.activation = nn.Sigmoid()

    def get_loss(self,predictions,labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        predictions = predictions.view(-1)
        labels = labels.float().view(-1)
        epsilon = 1e-8
        # 交叉熵
        loss = - labels * torch.log(predictions + epsilon) - (torch.tensor(1.0) - labels) * torch.log(torch.tensor(1.0) - predictions + epsilon)
        # 求均值, 并返回可以反传的loss
        # loss为一个实数
        loss = torch.mean(loss)
        return loss

    def forward(self,text,position,lables=None):
        hidden_layers,_ = self.bert(text,position,output_all_encoded_layers=True)

        sequence_output = hidden_layers[-1]

        cls_info = sequence_output[:,0]

        predictions = self.dense(cls_info)
        predictions = self.activation(predictions)

        if lables is not None:
            loss = self.get_loss(predictions,lables)
            return predictions,loss
        else:
            return predictions