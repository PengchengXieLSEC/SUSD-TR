import numpy as np
import math
import time
from utils.TRUNCATED_CG import TruncatedCG


class susd_optimizer:
    def __init__(self, f, alpha=0.5, k=1, max_iter=5000, sig=0, number_susd=0, number_tr=1, flag_map=1, flag_model=0):
        self.f = f
        self.alpha = alpha
        self.k = k
        self.max_iter = max_iter
        self.sig = sig         # 设置误差大小系数
        self.number_susd = number_susd  # 执行susd算法的次数（此时不执行tr）
        self.number_tr = number_tr  # 执行susd+tr的次数；susd_tr算法执行时，先进行number_susd次的susd，
                                    # 再进行number_tr次的susd_tr，如此循环直到达到最大迭代次数
        self.flag_map = flag_map  # 设置函数值到步长的映射：0-48， 对应伪代码第5行的\alpha_i(t)
        self.flag_model = flag_model   # 设置插值模型的种类---0:linear, 1:quadratic

        # self.c1=c1
        # self.c2=c2
        # self.c3=c3

    def step(self, x, nold, idminold):
        # perform a function evaluation of the current x estimate
        # fx_ = self.f(x)
        ###################################
        n = x.shape
        fx_ = np.zeros((1, n[1]))
        fx_ = fx_[0]
        for i in range(n[1]):
            fx_[i] = self.f(x[:, i])
        ###################################
        # compute the n direction; PCA
        Data = x.T
        U_mean = np.mean(x, axis=1)
        R_u = Data - U_mean
        Cov = np.matmul(R_u.T, R_u)
        w, v = np.linalg.eig(Cov)  # w为特征值 向量，v为特征向量矩阵
        idx = w.argsort()[::-1]  # 返回排序后的下标（小到大），再反向取索引--->结果为特征值从大到小排序的向量
        v = v[:, idx]
        v_1 = v[:, -1].real  # v_1为最小特征值对应的特征向量，即伪代码中的v_1

        # check if the direction reversed
        v_1 = np.sign(np.dot(v_1, nold)) * v_1
        nold = v_1

        #################################
        # 加噪声；fx对应伪代码中的\alpha_i
        fx = fx_ + self.sig * np.random.randn(x.shape[1])

        # perform the function transformation
        idmin, fmin = np.argmin(fx), min(fx)


        if self.flag_map == 0:
            fx = (fx-fmin)/max(fx-fmin)
        if self.flag_map == 1:
            fx = 1-np.exp(-self.k*(fx-fmin))
        if self.flag_map == 2:
            fx = np.power(fx-fmin, 3)/max(fx-fmin)**3
        if self.flag_map == 3:
            fx = np.power(fx-fmin, 5)/max(fx-fmin)**5
        if self.flag_map == 4:
            fx = (2*(fx-fmin)+np.sin(fx-fmin)) / \
                (2*(max(fx-fmin))+math.sin(max(fx-fmin)))
        if self.flag_map == 5:
            fx = (3.5*(fx-fmin)+np.cos(fx-fmin)) / \
                (3.5*(max(fx-fmin))+math.cos(max(fx-fmin)))
        if self.flag_map == 6:
            fx = np.log(1+(fx-fmin))/math.log(1+max(fx-fmin))
        if self.flag_map == 7:
            fx = (np.power(fx-fmin, 3)+np.sin(fx-fmin)) / \
                (max(fx-fmin)**3+math.sin(max(fx-fmin)))
        if self.flag_map == 8:
            fx = (np.power(fx-fmin, 3)+np.cos(fx-fmin)) / \
                (max(fx-fmin)**3+math.cos(max(fx-fmin)))
        if self.flag_map == 9:
            fx = (np.exp(fx-fmin))/math.exp(max(fx-fmin))
        if self.flag_map == 10:
            fx = (np.log(1+(fx-fmin))+(fx-fmin)) / \
                (math.log(1+(max(fx-fmin)))+max(fx-fmin))
        if self.flag_map == 11:
            fx = (np.power(fx-fmin, 4)+np.sin(fx-fmin)) / \
                (max(fx-fmin)**4+np.sin(max(fx-fmin)))
        if self.flag_map == 12:
            fx = (np.log(1+(fx-fmin))+np.power(fx-fmin, 3)) / \
                (math.log(1+(max(fx-fmin)))+max(fx-fmin)**3)
        if self.flag_map == 13:
            fx = (11*np.log(1+(fx-fmin))+np.sin(fx-fmin)) / \
                (11*math.log(1+(max(fx-fmin)))+math.sin(max(fx-fmin)))
        if self.flag_map == 14:
            fx = (15*np.log(1+(fx-fmin))+np.cos(fx-fmin)) / \
                (15*math.log(1+(max(fx-fmin)))+math.cos(max(fx-fmin)))
        if self.flag_map == 15:
            fx = np.arctan(fx-fmin)/math.atan(max(fx-fmin))
        if self.flag_map == 16:
            fx = np.power(np.log(1+(fx-fmin))/math.log(1+max(fx-fmin)), 1/2)
        if self.flag_map == 17:
            fx = np.log(1+np.log(1+(fx-fmin))/math.log(1+max(fx-fmin)))
        if self.flag_map == 18:
            fx = (np.power(fx-fmin, 1/3)+np.cos(fx-fmin)) / \
                (max(fx-fmin)**(1/3)+math.cos(max(fx-fmin)))
        if self.flag_map == 19:
            fx = (np.power(fx-fmin, 1/5)+np.cos(fx-fmin)) / \
                (max(fx-fmin)**(1/5)+math.cos(max(fx-fmin)))
        if self.flag_map == 20:
            fx = (np.power(fx-fmin, 1/3)+np.sin(fx-fmin)) / \
                (max(fx-fmin)**(1/3)+math.sin(max(fx-fmin)))
        if self.flag_map == 21:
            fx = (np.power(fx-fmin, 1/5)+np.sin(fx-fmin)) / \
                (max(fx-fmin)**(1/5)+math.sin(max(fx-fmin)))
        if self.flag_map == 22:
            fx = (np.power(fx-fmin, 1/7)+np.cos(fx-fmin)) / \
                (max(fx-fmin)**(1/7)+math.cos(max(fx-fmin)))
        if self.flag_map == 23:
            fx = (np.power(fx-fmin, 1/9)+np.cos(fx-fmin)) / \
                (max(fx-fmin)**(1/9)+math.cos(max(fx-fmin)))
        if self.flag_map == 24:
            fx = (np.power(fx-fmin, 1/7)+np.sin(fx-fmin)) / \
                (max(fx-fmin)**(1/7)+math.sin(max(fx-fmin)))
        if self.flag_map == 25:
            fx = (np.power(fx-fmin, 1/9)+np.sin(fx-fmin)) / \
                (max(fx-fmin)**(1/9)+math.sin(max(fx-fmin)))
        if self.flag_map == 26:
            fx = (np.power(fx-fmin, 1/11)+np.cos(fx-fmin)) / \
                (max(fx-fmin)**(1/11)+math.cos(max(fx-fmin)))
        if self.flag_map == 27:
            fx = (np.power(fx-fmin, 1/13)+np.cos(fx-fmin)) / \
                (max(fx-fmin)**(1/13)+math.cos(max(fx-fmin)))
        if self.flag_map == 28:
            fx = (np.power(fx-fmin, 1/11))/(max(fx-fmin)**(1/11))
        if self.flag_map == 29:
            fx = (np.power(fx-fmin, 1/13)+np.sin(fx-fmin)) / \
                (max(fx-fmin)**(1/13)+math.sin(max(fx-fmin)))
        if self.flag_map == 30:
            fx = (np.power(fx-fmin, 1/15)+np.cos(fx-fmin)) / \
                (max(fx-fmin)**(1/15)+math.cos(max(fx-fmin)))
        if self.flag_map == 31:
            fx = (np.power(fx-fmin, 1/17)+np.cos(fx-fmin)) / \
                (max(fx-fmin)**(1/17)+math.cos(max(fx-fmin)))
        if self.flag_map == 32:
            fx = (np.power(fx-fmin, 1/15)+np.sin(fx-fmin)) / \
                (max(fx-fmin)**(1/15)+math.sin(max(fx-fmin)))
        if self.flag_map == 33:
            fx = (np.power(fx-fmin, 1/17)+np.sin(fx-fmin)+(fx-fmin)) / \
                (max(fx-fmin)**(1/17)+math.sin(max(fx-fmin))+max(fx-fmin))
        if self.flag_map == 34:
            fx = (np.power(fx-fmin, 1/17)+np.sin(fx-fmin)+(fx-fmin)) / \
                (max(fx-fmin)**(1/17)+math.sin(max(fx-fmin))+max(fx-fmin))
        if self.flag_map == 35:
            fx = (1-np.exp(-np.power((fx-fmin), 5))) / \
                (1-math.exp(-max(fx-fmin)**5))
        if self.flag_map == 36:
            fx = (np.log(1+np.power(fx-fmin, 5)))/(math.log(1+max(fx-fmin)**5))
        if self.flag_map == 37:
            fx = np.power((1-np.exp(-(fx-fmin))), 5) / \
                (1-math.exp(-max(fx-fmin)))**5
        if self.flag_map == 38:
            fx = np.arctan(np.power(fx-fmin, 5))/math.atan(max(fx-fmin)**5)
        if self.flag_map == 39:
            fx = np.power(np.arctan(fx-fmin), 5)/math.atan(max(fx-fmin))**5
        if self.flag_map == 40:
            fx = (2*np.arctan(fx-fmin)+np.sin(np.arctan(fx-fmin)))/2 * \
                math.atan(max(fx-fmin))+math.sin(math.atan(max(fx-fmin)))
        if self.flag_map == 41:
            fx = np.log(1+np.arctan(fx-fmin)) / \
                math.log(1+math.atan(max(fx-fmin)))
        if self.flag_map == 42:
            fx = (2*np.power(fx-fmin, 1/11)+np.sin(np.power(fx-fmin, 1/11)))/(2*max(fx-fmin)**(1/11)+math.sin(max(fx-fmin)**(1/11)))
        if self.flag_map == 43:
            fx = np.power(2*(fx-fmin)+np.sin(fx-fmin), 1/11) / \
                (2*max(fx-fmin)+math.sin(max(fx-fmin)))**(1/11)
        if self.flag_map == 44:
            fx = np.log(1+np.power(fx-fmin, 1/11)) / \
                math.log(1+max(fx-fmin)**(1/11))
        if self.flag_map == 45:
            fx = np.power(np.log(1+(fx-fmin)), 1/11) / \
                math.log(1+max(fx-fmin))**(1/11)
        if self.flag_map == 46:
            fx = np.power(np.log(1+2*(fx-fmin)+np.sin(fx-fmin)), 1/11) / \
                math.log(1+2*max(fx-fmin)+math.sin(max(fx-fmin)))**(1/11)
        if self.flag_map == 47:
            fx = np.log(1+np.power(2*(fx-fmin)+np.sin(fx-fmin), 1/11)) / \
                math.log(1+(2*max(fx-fmin)+math.sin(max(fx-fmin)))**1/11)
        if self.flag_map == 48:
            fx = np.power(2*np.log(1+(fx-fmin))+np.sin(fx-fmin), 1/11) / \
                (2*math.log(1+max(fx-fmin))+math.sin(max(fx-fmin)))**(1/11)

        # check if the same index is still the minimum
        stationary = (idmin == idminold)
        idminold = idmin

        # step using the SUSD method
        x = x + self.alpha * np.outer(v_1, fx)  # 求外积，v_1.T * fx

        # return if the same minimum was found
        return stationary, x, min(fx_), nold, idminold

    def TR_2dim(self, x):  # 2维时的信赖域子方法
        # fx = self.f(x)
        ###################################
        n = x.shape
        fx = np.zeros((1, n[1]))
        fx = fx[0]
        for i in range(n[1]):
            fx[i] = self.f(x[:, i])
        ###################################

        idmin = np.argmin(fx)
        idmax = np.argmax(fx)
        xx = np.delete(x, idmax, axis=1)
        fxx = np.delete(fx, idmax)
        delta = 1
        A = (xx[1, 1]-xx[1, 0])*(fxx[2]-fxx[0]) - \
            (xx[1, 2]-xx[1, 0])*(fxx[1]-fxx[0])
        B = (xx[0, 2]-xx[0, 0])*(fxx[1]-fxx[0]) - \
            (xx[0, 1]-xx[0, 0])*(fxx[2]-fxx[0])
        C = (xx[0, 1]-xx[0, 0])*(xx[1, 2]-xx[1, 0]) - \
            (xx[0, 2]-xx[0, 0])*(xx[1, 1]-xx[1, 0])
        D = -(A*xx[0, 0]+B*xx[1, 0]+C*fxx[0])
        a = -A/C
        b = -B/C
        c = -D/C
        x1 = -b*delta/(math.sqrt(a*a+b*b))+x[0, idmin]
        y1 = -a*delta/(math.sqrt(a*a+b*b))+x[1, idmin]
        x[0, idmax] = x1
        x[1, idmax] = y1
        idmin = np.argmin(fx)
        fmin = fx[idmin]
        idminold = idmin
        return x, fmin, idminold

    def TR(self, x, delta):  # 更一般形式的信赖域子方法
        if self.flag_model == 0:  # 一次模型
            # fx = self.f(x)
            ###################################
            n = x.shape
            fx = np.zeros((1, n[1]))
            fx = fx[0]
            for i in range(n[1]):
                fx[i] = self.f(x[:, i])
            ###################################
            idmin = np.argmin(fx)
            idmax = np.argmax(fx)
            xx = np.delete(x, idmax, axis=1)
            fxx = np.delete(fx, idmax)
            Fxx = np.matrix(fxx)
            # delta=0.6
            xx = xx.T
            n = xx.shape
            xx = np.c_[xx, np.ones((n[0], 1))]
            A = np.matmul(np.linalg.inv(xx), Fxx.T)
            B = A[0:n[0]-1, 0]
            C = np.matrix(x[:, idmin])
            x_new = -delta/math.sqrt(np.matmul(B.T, B))*B+C.T

            x = x.T
            x[idmax] = x_new.T
            x = x.T

        #################################
        if self.flag_model == 1:  # 二次模型
            #fx = self.f(x)
            ###################################
            n = x.shape
            fx = np.zeros((1, n[1]))
            fx = fx[0]
            for i in range(n[1]):
                fx[i] = self.f(x[:, i])
            ###################################

            idmin = np.argmin(fx)
            idmax = np.argmax(fx)
            xx = np.delete(x, idmax, axis=1)
            fxx = np.delete(fx, idmax)
            Fxx = np.matrix(fxx)
            # delta=0.6
            n = xx.shape
            AA = np.zeros((n[1], n[1]))

            for i in range(n[1]):
                xx[:, i] = xx[:, i]-x[:, idmin]
            xx = xx.T
            for i in range(n[1]):
                A = []
                for j in range(n[0]):
                    for k in range(n[0]):
                        if j == k:
                            A = np.append(A, xx[i, j]**2)
                        elif j < k:
                            A = np.append(A, 2*xx[i, j]*xx[i, k])
                AA[i][0:len(A)] = A

            P = np.matrix(xx)
            AA[:, len(A):len(A)+n[0]] = P
            AA[:, -1] = np.ones((n[1]))

            B = np.matmul(np.linalg.inv(AA), Fxx.T)
            H = np.zeros((n[0], n[0]))
            index = 0
            for j in range(n[0]):
                L = B[index:index+n[0]-j, 0]
                H[j, j:n[0]] = L.T
                index = index+n[0]-j
            for j in range(n[0]):
                for k in range(n[0]):
                    if k > j:
                        H[j, k] = H[j, k]/2
                    if k < j:
                        H[j, k] = H[k, j]
            b = B[index:index+n[0]]
            c = B[-1]

            kwargs = {'radius': delta, 'absol': 1.0e-8, 'reltol': 1.0e-6,
                      'maxiter': 50, 'prec': 1.0e-6*np.ones((n[0], 1))}
            C = TruncatedCG(b, 1/2*H, **kwargs)
            S = C.Solve(**kwargs)
            C = np.matrix(x[:, idmin])

            x = x.T
            P = np.matrix(x[idmax, :])
            x[idmax, :] = x[idmin, :]+np.array(S.T)
            x = x.T
        #################################
        #################################
        x = x.T
        Ared_k = self.f(x[idmin])-self.f(x[idmax])
        if self.flag_model == 0:  # 一次模型
            Pred_k = np.matmul(x[idmin], B)-np.matmul(x[idmax], B)
        if self.flag_model == 1:  # 二次模型
            xmin = x[idmin]
            xmax = x[idmax]
            Hx = np.matmul(H, xmin.T)
            Hy = np.matmul(H, xmax.T)
            Pred_k = np.matmul(xmin, Hx)+np.matmul(xmin, b) - \
                (np.matmul(xmax, Hy)+np.matmul(xmax, b))

        r_k = Ared_k/Pred_k
        if r_k[0][0] <= 1/4:
            delta = 1/4*delta
        elif (1/4 < r_k[0][0]) & (r_k[0][0] < 3/4):
            delta = 2*delta
        else:
            delta = delta
        x = x.T
        #################################

        idmin = np.argmin(fx)
        fmin = fx[idmin]
        idminold = idmin
        if delta < 0.00000001:
            delta = 0.00000001

        return x, fmin, idminold, delta


    def search(self, x0, term_len=200, return_hist=True):
        # setup SUSD search
        delta = 1
        count = 0
        pfmin = float('Inf')  # 正无穷
        nold = np.random.rand(x0.shape[0])
        idminold = -1
        x = x0.copy()
        if return_hist:
            hist = {}

        non_parallel_time = 0
        # begin search
        for it in range(self.max_iter):
            # perform an SUSD step

            # 计算不能并行的时间
            # 只有在启动信赖域子方法的时候才去计算不能并行的时间，即伪代码第8行，执行信赖域子方法的时间
            # 如果不执行信赖域子方法，则不能并行的时间为 0

            # number_susd ==0，即所有迭代次数都执行tr
            if self.number_susd == 0:
                stationary, x, fmin, nold, idminold = self.step(x, nold.copy(), idminold)
                start_non_parallel = time.time()
                x, fmin, idminold, delta = self.TR(x, delta)
                end_non_parallal = time.time()
                non_parallel_time += (end_non_parallal - start_non_parallel)
                # print('只执行susd-tr')
            # number_tr == 0，即所有迭代次数都只执行susd
            elif self.number_tr == 0:
                stationary, x, fmin, nold, idminold = self.step(x, nold.copy(), idminold)
                # print('只执行susd')

            # 仅执行susd的迭代次数it
            elif it % (self.number_susd + self.number_tr) in [num for num in range(self.number_susd)]:
                stationary, x, fmin, nold, idminold = self.step(x, nold.copy(), idminold)
                # print('执行susd')
            # 执行susd+tr即susd_tr的的迭代次数it
            # 在每一次迭代末尾启动信赖域子方法
            elif it % (self.number_susd + self.number_tr) in [num for num in range(self.number_susd, self.number_susd + self.number_tr)]:
                stationary, x, fmin, nold, idminold = self.step(x, nold.copy(), idminold)
                start_non_parallel = time.time()
                x, fmin, idminold, delta = self.TR(x, delta)
                end_non_parallal = time.time()
                non_parallel_time += (end_non_parallal - start_non_parallel)
                # print('执行susd-tr')

            # store iteration history
            if return_hist:
                hist[it] = (x, x[:, idminold], min(fmin, pfmin))
            pfmin = min(fmin, pfmin)

            # check termination condition
            if stationary:
                count += 1
            else:
                count = 0

            if count == term_len:
                if return_hist:
                    return it, hist, non_parallel_time
                else:
                    return it, (x[:, idminold], fmin), non_parallel_time

        # search did not converge
        print("Warning, max search limit exceeded")
        if return_hist:
            return it, hist, non_parallel_time
        else:
            return it, (x[:, idminold], fmin), non_parallel_time
