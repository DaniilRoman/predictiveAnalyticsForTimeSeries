from multiprocessing import Pool, Process
import time

from src.processing.DataHolder import DataHolder


# def func(arg="check"):
#     time.sleep(1)
#     print(arg)



# if __name__ == '__main__':
#     pass
    # p = Process(target=func, args=("test",))
    # p.start()
    # p.join() # означает что мы присоединяем процесс p к мейн процессу и получается что мы выполняем эо последоватеьно то есть дожидаемся завершения работы процесса p

    # dataHolder = DataHolder()
    # p = Process(target=dataHolder.storeNewValue)
    # p.start()










#
# import time
# from multiprocessing import Process, Value, Lock
#
# def func(val, lock):
#     for i in range(50):
#         time.sleep(0.01)
#         with lock:
#             val.value += 1
#     print(val.value)
#
# if __name__ == '__main__':
#     v = Value('i', 0)
#     lock = Lock()
#     v.value += 1
#     procs = [Process(target=func, args=(v, lock)) for i in range(10)]
#
#     for p in procs: p.start()
#     # for p in procs: p.join()
#
#     # print(v.value)





#
# from multiprocessing import Process, Pipe
# import time
#
# def reader_proc(pipe):
#     p_output, p_input = pipe
#     p_input.close()
#     while p_output.poll():
#         msg = p_output.recv()
#         # print(msg)
#
# def writer(count, p_input):
#     for ii in range(0, count):
#         p_input.send(ii)
#
# if __name__=='__main__':
#     for count in [10]:
#         p_output, p_input = Pipe()
#         reader_p = Process(target=reader_proc, args=((p_output, p_input),))
#         reader_p.daemon = True
#         reader_p.start()
#         p_output.close()
#
#         _start = time.time()
#         writer(count, p_input) # Send a lot of stuff to reader_proc()
#         p_input.close()
#         reader_p.join()
#         print("Sending {0} numbers to Pipe() took {1} seconds".format(count,
#             (time.time() - _start)))





import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

diabetes = datasets.load_diabetes()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)
model = LinearRegression()
# print(X_train)
# print(X_train.shape)
X_train = X_train[:, 0].reshape(-1, 1)
X_test = X_test[:, 0].reshape(-1, 1)
# print(X_train)
# print(X_train.shape)
# print(y_train)
# print(y_train.shape)
# 2. Use fit
model.fit(X_train, y_train)
# 3. Check the score
print(X_test)
print(X_test.shape)
predict = model.predict(X_test)

# print(predict)
# print(predict.shape)





