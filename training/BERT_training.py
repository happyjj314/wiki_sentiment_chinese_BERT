from torch.utils.data import DataLoader

from dataset.wikiDataset import BERTDataset
from models.bert_model import *
import tqdm
import pandas as pd
import numpy as np
import os

config = {}
config["train_corpus_path"] = "./wiki_dataset/train_wiki.txt"
config["test_corpus_path"] = "./wiki_dataset/test_wiki.txt"
config["word2idx_path"] = "../corpus/bert_word2idx.json"
config["output_path"] = "../output_wiki_bert"
config["batch_size"] = 16  #8g显存已经极限了 就只能16了
config["max_seq_len"] = 200
config["vocab_size"] = 32162
config["lr"] = 2e-6
config["num_workers"] = 0

class Pretrainer:
    def __init__(self,bert_model,vocab_size,max_len,batch_size,lr,with_code = True,):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        # 初始化bert训练模型
        bertconfig = BertConfig(vocab_size=config["vocab_size"])
        #初始化参数
        self.bert_model = bert_model(config=bertconfig)
        self.bert_model.to(self.device)

        train_dataset = BERTDataset(corpus_path=config["train_corpus_path"],
                                    word2idx_path=config["word2idx_path"],
                                    seq_len=self.max_len,
                                    hidden_dim=bertconfig.hidden_size
                                    )
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers= 0,
                                           collate_fn= lambda x:x)

        test_dataset = BERTDataset(corpus_path=config["test_corpus_path"],
                                    word2idx_path=config["word2idx_path"],
                                    seq_len=self.max_len,
                                    hidden_dim=bertconfig.hidden_size
                                    )
        self.test_dataloader = DataLoader(test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=0,
                                           collate_fn=lambda x: x)

        self.positional_enc = self.init_positional_encoding(hidden_dim=bertconfig.hidden_size,max_seq_len=self.max_len)
        self.positional_enc = torch.unsqueeze(self.positional_enc,dim=0)

        self.optimizer = torch.optim.Adam(list(self.bert_model.parameters()),lr=self.lr)



    def init_positional_encoding(self, hidden_dim, max_seq_len):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / hidden_dim) for i in range(hidden_dim)]
            if pos != 0 else np.zeros(hidden_dim) for pos in range(max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def padding(self,data):
        # 动态加载，每次加载batch中最长的长度
        bert_input = [i["bert_input"] for i in data]
        bert_label = [i["bert_label"] for i in data]
        segment_label = [i["segment_label"] for i in data]
        bert_input = torch.nn.utils.rnn.pad_sequence(bert_input, batch_first=True)
        bert_label = torch.nn.utils.rnn.pad_sequence(bert_label, batch_first=True)
        segment_label = torch.nn.utils.rnn.pad_sequence(segment_label, batch_first=True)
        is_next = torch.cat([i["is_next"] for i in data])
        return {"bert_input": bert_input,
                "bert_label": bert_label,
                "segment_label": segment_label,
                "is_next": is_next}

    def mask_acc(self,predictions,labels):
        predictions = torch.argmax(predictions,dim=-1,keepdim=False)
        mask = (labels > 0).to(self.device)
        acc = torch.sum((predictions == labels) * mask).float()
        acc /= (torch.sum(mask).float() + 1e-8)
        return acc.item()

    def compute_loss(self, predictions, labels, num_class=2, ignore_index=None):
        if ignore_index is None:
            loss_func = CrossEntropyLoss()
        else:
            loss_func = CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(predictions.view(-1, num_class), labels.view(-1))

    def train(self,epoch):
        self.bert_model.train()
        self.iteration(epoch,self.train_dataloader)
    def test(self,epoch):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch,self.test_dataloader,train=False)

    def iteration(self,epoch,data_loader,train=True,df_path = "../output_wiki_bert/df_wiki_log.pickle"):
        if not os.path.isfile(df_path):
            df = pd.DataFrame(columns=["epoch", "train_next_sen_loss", "train_mlm_loss",
                                       "train_next_sen_acc", "train_mlm_acc",
                                       "test_next_sen_loss", "test_mlm_loss",
                                       "test_next_sen_acc", "test_mlm_acc"
                                       ])
            df.to_pickle(df_path)

        str_code = 'train' if train else 'test'
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="%s_part, %d epoch"%(str_code,epoch),
                              total=len(data_loader))

        total_next_sen_loss = 0
        total_mlm_loss = 0
        total_next_sen_acc = 0
        total_mlm_acc = 0
        total_element = 0

        for i ,data in data_iter:
            data = self.padding(data)
            data = {key: value.to(self.device) for key, value in data.items()}
            positional_enc = self.positional_enc[:, :data["bert_input"].size()[-1], :].to(self.device)

            pre_mask,next_sen_pre = self.bert_model.forward(input_ids=data["bert_input"],
                                                                positional_enc=positional_enc,
                                                                token_type_ids=data["segment_label"])
            mask_acc = self.mask_acc(pre_mask,data["bert_label"])
            next_sen_acc = next_sen_pre.argmax(dim=-1, keepdim=False).eq(data["is_next"]).sum().item()
            mask_loss = self.compute_loss(pre_mask,data["bert_label"],self.vocab_size,ignore_index=0)
            next_sen_loss = self.compute_loss(next_sen_pre,data["is_next"])
            loss = mask_loss + next_sen_loss

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_next_sen_loss += next_sen_loss.item()
            total_mlm_loss += mask_loss.item()
            total_next_sen_acc += next_sen_acc
            total_mlm_acc += mask_acc
            total_element += data["is_next"].nelement()


            if train:
                log_dic = {
                    "epoch": epoch,
                   "train_next_sen_loss": total_next_sen_loss / (i + 1),
                   "train_mlm_loss": total_mlm_loss / (i + 1),
                   "train_next_sen_acc": total_next_sen_acc / total_element,
                   "train_mlm_acc": total_mlm_acc / (i + 1),
                   "test_next_sen_loss": 0, "test_mlm_loss": 0,
                   "test_next_sen_acc": 0, "test_mlm_acc": 0
                }

            else:
                log_dic = {
                    "epoch": epoch,
                   "test_next_sen_loss": total_next_sen_loss / (i + 1),
                   "test_mlm_loss": total_mlm_loss / (i + 1),
                   "test_next_sen_acc": total_next_sen_acc / total_element,
                   "test_mlm_acc": total_mlm_acc / (i + 1),
                   "train_next_sen_loss": 0, "train_mlm_loss": 0,
                   "train_next_sen_acc": 0, "train_mlm_acc": 0
                }

            if i % 10 == 0:
                # 10轮打印一下 loss这些
                data_iter.write(str({k: v for k, v in log_dic.items() if v != 0 and k != "epoch"}))

        if train:
            df = pd.read_pickle(df_path)
            df = df.append([log_dic])
            df.reset_index(inplace=True, drop=True)
            df.to_pickle(df_path)
        else:
            log_dic = {k: v for k, v in log_dic.items() if v != 0 and k != "epoch"}
            df = pd.read_pickle(df_path)
            df.reset_index(inplace=True, drop=True)
            for k, v in log_dic.items():
                df.at[epoch, k] = v
            df.to_pickle(df_path)
            return float(log_dic["test_next_sen_loss"])+float(log_dic["test_mlm_loss"])


    def save_state_dict(self, model, epoch, dir_path="../output", file_path="bert.model"):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        save_path = dir_path+ "/" + file_path + ".epoch.{}".format(str(epoch))
        model.to("cpu")
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print("{} saved!".format(save_path))
        model.to(self.device)

    def load_model(self, model, dir_path="../output_wiki_bert/bert.model.epoch.10"):
        # 加载模型
        model.load_state_dict(dir_path, strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded for training!".format(dir_path))

if __name__ == '__main__':
    trainer = Pretrainer(BertForPreTraining,
                         vocab_size=config["vocab_size"],
                         max_len=config["max_seq_len"],
                         batch_size=config["batch_size"],
                         lr=config["lr"],
                         with_code=True)
    for epoch in range(0,10):
        trainer.train(epoch)
        trainer.save_state_dict(trainer.bert_model,epoch,dir_path=config["output_path"],file_path="bert.model")
        trainer.test(epoch)