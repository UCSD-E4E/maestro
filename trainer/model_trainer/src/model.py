import os
import time
print("importing in model", flush=True)
start = time.time()

from torch import nn, Tensor, load, save
from torch.optim import Adam
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

end = time.time()
print("import time: ", end - start, flush=True)

# This takes too long!!!!
from timm import create_model

end = time.time()
print("import timm: ", end - start, flush=True)



class Trainer():
    def __init__(self, conf, data):
        self.conf, self.data = conf, data
        self.dataloader = None
        self.model = None

    def build(self):
        dataset = DeepFishDataset(self.data, self.conf["data_path"])
        self.dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        self.model = Model(self.conf).cuda()

    def train(self):
        # https://machinelearningmastery.com/creating-a-training-loop-for-pytorch-models/
        # Code used as template
       
        loss_fn = self.model.loss()
        optimizer = self.model.optimizer()
        loss_sum = 0
        count = 0
        self.model.train()
        for (input, label) in self.dataloader:
                
                
                count += 1

                optimizer.zero_grad()
               
                pred = self.model(input.cuda())
                
                #print(pred, label)
                loss = loss_fn(pred, label.cuda())
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()#.cpu()
                
        adv_loss = loss_sum/count
        #print(adv_loss)
        self.model.save()
        return adv_loss

class DeepFishDataset(Dataset):
    def __init__(self, data, data_path):
        super()
        self.data = data
        self.data_path = data_path
        #TODO: pull data from file_path into local container
        #Take advantage of potential parrellism

    def __len__(self):    
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        file_path = os.path.join(self.data_path, data["file_path"])
        label = Tensor([data["label"]])
        #print(label, data["label"], type(data["label"]))
        image = read_image(file_path).float()
        return image, label

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.model = create_model(
            cfg["model_name"],
            pretrained=(not os.path.exists(cfg["model_checkpoint"])),
            num_classes=1,
        )

        self.checkpoint = None
        self.optimizer_obj = None
        if cfg["model_checkpoint"] != "" and os.path.exists(cfg["model_checkpoint"]):
            # Hypothesis: Model loads can be affected by processes accessing the same
            # File, so here is a system to wait for another process to finsh
            # Note: Written after getting a "file read failed" error but did not
            # obsreve file corruption and could successfully load the failed file
            # 30 mins after a failed run. 
            for i in range(10):
                if i > 8:
                    print("HEY THE FILE IS ACTUALLY CORRUPTED!!!")
                    return
                try:
                    self.checkpoint = load(cfg["model_checkpoint"])
                    self.model.load_state_dict(self.checkpoint['model_state_dict'])
                except Exception as e:
                    print(e, "retrying:", i)
                    time.sleep(15)
                    continue
                break

    def forward(self, images):
        return self.model(images)

    def save(self):
        print(self.cfg["model_checkpoint"], flush=True)
        save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer_obj.state_dict(),
            }, self.cfg["model_checkpoint"])

    def optimizer(self):
        optimizer = Adam(self.model.parameters(), lr=self.cfg["learning_rate"])
        if self.checkpoint is not None:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.optimizer_obj = optimizer
        return optimizer

    def loss(self):
        return nn.BCEWithLogitsLoss(reduction='mean')