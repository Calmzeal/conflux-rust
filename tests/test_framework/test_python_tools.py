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

if car:
    print("Yes")
else:
    print(bool({}))
print(bool({"s":2}))

a =list(range(10))


print(a)
print(list(range(0,5,2)).remove(4))
print(list(range(1,5,2)))
count = 0
count+=1
x,_,_ = (2,3,5)
print(x)
print(int("0x3e",0))

import queue
q = queue.Queue()
print(q.qsize())
q.put(1)
q.put(2)
q.get()
print(q.qsize())
print(set([1,2,3,3,4]))
print(1/0.8)