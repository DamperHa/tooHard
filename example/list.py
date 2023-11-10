bicycles = ['trek','cannondale','redline','specialized']
print(bicycles)

print(bicycles[0].title())

# 末尾添加元素
bicycles.append("fanzhihao")
print(bicycles)


# 列表中插入元素
bicycles.insert(0, "ok")
print(bicycles)


del bicycles[0]
print(bicycles)

# 排序
bicycles.sort()
print(bicycles)
