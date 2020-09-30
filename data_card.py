data_card={
    "type": "image classification",
    "file_format": ".jpg image files",
    "sources":{
        "name":"NSFW - not-safe-for-work",
        "Url":"https://www.kaggle.com/omeret/not-safe-for-work",
    },
    'license':'GPL2',
    "file_location":"https://www.kaggle.com/omeret/not-safe-for-work",
    "use_cases": "Image classification for content moderation",
    "file_columns": None,
    "file_size":"19.59 GB",
    "total_data_samples":"102.5K",
    "labels":['NSFW', 'SFW'],
    "data_distribution":{
        'train':{'nsfw':'63.0k','sfw':'35.5k'},
        'test':{'nsfw':'2k','sfw':'2k'},
    }
}
import json
with open('data_card.json', 'w') as fp:
    json.dump(data_card, fp)