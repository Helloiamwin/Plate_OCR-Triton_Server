import pymongo


user = "ai_to_product"
password = "ai_to_product"
hostname = "192.168.0.118"
port = "27017"

url = f"mongodb://{user}:{password}@{hostname}:{port}"
client = pymongo.MongoClient(url)


# database
# Table --> Collection
# Row   --> Document

database = client["face_db"]
faces = database["face"]

# mylist = [
#   { "name": "Amy", "address": "Apple st 652"},
#   { "name": "Hannah", "address": "Mountain 21"},
#   { "name": "Michael", "address": "Valley 345"},
#   { "name": "Sandy", "address": "Ocean blvd 2"},
#   { "name": "Betty", "address": "Green Grass 1"},
#   { "name": "Richard", "address": "Sky st 331"},
#   { "name": "Susan", "address": "One way 98"},
#   { "name": "Vicky", "address": "Yellow Garden 2"},
#   { "name": "Ben", "address": "Park Lane 38"},
#   { "name": "William", "address": "Central st 954"},
#   { "name": "Chuck", "address": "Main Road 989"},
#   { "name": "Viola", "address": "Sideway 1633"}
# ]

# x = faces.insert_many(mylist)

for x in faces.find():
  print(x)
print("--------------------------------------------")

myquery = { "address": "Valley 345" }
newvalues = { "$set": { "address": "Canyon 123" } }

faces.update_one(myquery, newvalues)
#print "customers" after the update:
for x in faces.find():
  print(x)