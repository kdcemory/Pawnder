import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

collections = db.collections()
for col in collections:
    print(f'Found collection: {col.id}')
