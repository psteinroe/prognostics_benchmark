from pymongo import MongoClient
from tqdm import tqdm

if __name__ == '__main__':
    CON_STRING = 'localhost:27017'

    client = MongoClient(CON_STRING)
    db = client.harddrive_preprocessing

    n_models = len(db.list_collection_names())
    n_equis = 0
    n_failures = 0
    n_records = 0
    attributes = set()
    for model_id in tqdm(db.list_collection_names(), total=len(db.list_collection_names())):
        # number of equipments
        n_equis = n_equis + len(db[model_id].find().distinct('serial_number'))
        # number of failures
        n_failures = n_failures + len(list(db[model_id].find({'failure': 1})))
        # of records
        n_records = n_records + db[model_id].count_documents({})

    print("Number of Models: {}".format(n_models))
    print("Number of Equipments: {}".format(n_equis))
    print("Number of Failures: {}".format(n_failures))
    print("Number of Records: {}".format(n_records))
