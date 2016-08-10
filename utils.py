import numpy as np


def save_model_parameters_theano(outfile, model):
    U, V, W,W1 = model.U.get_value(), model.V.get_value(), model.W.get_value(),model.W1.get_value()
    np.savez(outfile, U=U, V=V, W=W,W1=W1)
    print "Saved models parameters to %s." % outfile


def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W, W1 = npzfile["U"], npzfile["V"], npzfile["W"], npzfile["W1"]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    model.W1.set_value(W1)
    print "Loaded models parameters from %s. succuse!" % (path)


def cat_map(train_data_code,train_label,pre_code,pre_label,N=500):
    size = pre_code.shape[0]
    sim = np.dot(train_data_code, np.transpose(pre_code))
    inx = np.argsort(-sim, 0)
    inx = inx[:N]
    lab = np.array(train_label)
    pre_label = np.array(pre_label)
    nn_lab = lab[inx]
    apall = 0
    for i in range(size):
        x=0
        p=0
        new_label = nn_lab[:,i]
        for j in range(N):
            if new_label[j]==pre_label[i]:
                x = x + 1
                p = p + x*1./(j+1)
        if x:
            apall += p/x
    return apall/size


def get_pre(train_data_code,train_label,pre_code,pre_label,N=1000):
    size = pre_code.shape[0]
    sim = np.dot(train_data_code,np.transpose(pre_code))
    inx = np.argsort(-sim,0)
    inx = inx[:N]
    lab = np.array(train_label)
    pre_label = np.array(pre_label)
    nn_lab = lab[inx]
    pre = 0.
    for i in range(size):
        pre = pre + np.sum(nn_lab[:,i]==pre_label[i])*1./N


def num2bit(data):
    if len(data.shape) == 1:
        data = np.array([data])
    data = data.transpose()
    ret = []
    for i in data:
        t = []
        for j in i:
            for k in xrange(3):
                #fetch each bit from right to left
                a = (j>>(2-k))&1
                if a==0:
                    a= -1.

                t.append(a)
        ret.append(t)
    return np.array(ret)



