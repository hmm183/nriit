from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['medical_data']
collection = db['medical_records']

# Test a simple query to check if MongoDB is working
result = collection.find_one({"question": "What is diabetes?"})
print(result)
