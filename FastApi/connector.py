from fastapi import FastAPI
from typing import Optional

app = FastAPI()
animals = []
inventory = {
1:{
    "Name" : "Milk",
    "Price" : 2.99
}
}

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.get("/animals/{query}")
def animal(query: str):
    animals.append(query)
    return {"animals": animals, "count": len(animals)}

@app.get("/get-items/{id_items}")
def items(id_items: int):
    return inventory[id_items]

@app.get("/get-by-name")
def name(*, name: Optional[str] = None) :
    for item_name in inventory:
        if inventory[item_name]["Name"] == name:
            return inventory[item_name]
    return{"Data is invalid"}
