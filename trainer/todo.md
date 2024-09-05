- model class needs to be able to run a given pytorch model
- instantly train 1 iteration given set of data points
- dataset to act as transformer 


```
# Method that retruns dataloader, model
builder(train_info):
    for datapoint in train_info["data"]:
        datapoint["data"]
        datapoint["label"]

    => create a dataset => dataloader

    model(train_info["model_conf"])
    return dataloader, model

-------------
# Generic that returns loss and model from training
trainer(dataloader, model):
    model.optimizer
    training loop here
    return cirterion, model

-------------

main(train_info):
    dataloader, model = builder(train_info)
    cirterion, model = trainer(dataloader, model)
    model.save()
    return cirterion

```





Trainer
- Desc
    Object to handle download of data, model/dataset creatation and training of model
- Method
    - pull_data 
        consider it in init?
    - create dataloader
    - create model


Model
- Instance Vars

- Methods
    - Train method
    -