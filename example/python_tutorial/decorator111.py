def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")

    return wrapper


# 装饰器类似于go中函数的wrapper，函数是一等公民，函数传入到函数中，丰富其功能
@my_decorator
def say_hello():
    print("Hello!")

say_hello()