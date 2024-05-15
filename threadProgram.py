import time
from threading import *

# class first(Thread):
#     def run(self):
#         for i in range(50):
#             print("mygreet1")
#             time.sleep(1)

# class second(Thread):
#     def run(self):
#         for i in range(50):
#             print("mygreet2\n")
#             time.sleep(1)

# inst1 = first()
# inst2 = second()
# inst1.start()
# inst2.start()

def function1():
    print("hello1")
    time.sleep(5)

def function2():
    print("hello2")
    time.sleep(3)

# function1()
# function2()
func1Thread = Thread(target=function1)
func2Thread = Thread(target=function2)
func1Thread.daemon = True
func2Thread.daemon = True
func1Thread.start()
func2Thread.start()

