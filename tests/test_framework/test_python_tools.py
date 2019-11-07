car = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}

x = car.get("brand")
if x is None:
    print("???")
print(x)
print(car.items())
print(car.values())
car.update({"brand":"dfs"})
print(car)