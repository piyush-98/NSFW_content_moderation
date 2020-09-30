model_card={
    "type": "image classification",
    "prediction_type":"binary classification",
    "dataset_used": "https://www.kaggle.com/omeret/not-safe-for-work",
    "output_labels": {'0':'nsfw','1':'sfw'},
    "model_unit": "CNN",
    "used_pretrained_embeddings": 'False',
    'pretrained_embedding':None,
    "val_accuracy": '90.1',
    "usage": "image moderation by detecting NSFW content",
    "limitations":"",
    "model_size": "4.89 Mb",
    "input_data_type": "image vector with a shape of (100,100,3)",
    "backend": "Tensorflow Keras",
    "trained_on": "NVidia K80 GPU",
    "train_test_split":{'train':'0.8','validation':'0.2'},
    "classification_report":{
        '0':{'precision':'0.85','recall':'0.91','f1_score':'0.88'},
        '1':{'precision':'0.91','recall':'0.84','f1_score':'0.87'},
        'overall':{'precision':'0.88','recall':'0.88','f1_score':'0.88'}
    }
}
import json
with open('model_card.json', 'w') as fp:
    json.dump(model_card, fp)
