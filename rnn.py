import numpy as np
import theano as theano
import theano.tensor as T
import theano.typed_list
import operator

class RNNTheano:

    def __init__(self):

        W1 = np.random.uniform(-np.sqrt(1./5120), np.sqrt(1./5120), (512, 128))
        U = np.random.uniform(-np.sqrt(1./128), np.sqrt(1./128), (128, 64))
        V = np.random.uniform(-np.sqrt(1./64), np.sqrt(1./64), (64, 8))
        W = np.random.uniform(-np.sqrt(1./64), np.sqrt(1./64), (64, 64))

        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W1 = theano.shared(name='W', value=W1.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.L = 10
        self.learning_rate = 10e-3 * 0.5
        self.batch_size = 5000
        self.lamd3 =  10e-6
        self.__rnn_build__()

    def __rnn_build__(self):
        U, V, W1, W = self.U, self.V, self.W1,self.W
        x = T.matrix('x')
        h1 = T.tanh(T.dot(x, W1))

        def forward_prop_step(x_t, s_t_prev, U, V):
            s_t = T.tanh(T.dot(x_t,U)+ T.dot(s_t_prev,W))
            h3 = T.tanh(T.dot(s_t,V))
            B = T.switch(T.lt(h3, 0.), -1., 1.)
            loss = 0.5 * T.sum((B-h3) ** 2)
            mx=T.argmax(h3,axis=1)
            return [s_t,mx,loss]

        def forward_first_step(x_t, U, V):
            s_t = T.tanh(T.dot(x_t,U))
            h3 = T.tanh(T.dot(s_t,V))
            B = T.switch(T.lt(h3, 0.), -1., 1.)
            loss = 0.5 * T.sum((B-h3) ** 2) / self.batch_size
            return [s_t,B,loss]

        [last,_,f_loss] = forward_first_step(h1,U,V)
        for i in xrange(self.L-1):
            [pre,_,lossi] = forward_prop_step(h1,last,U,V)
            last = pre
            f_loss += lossi
        f_loss += 0.5*self.lamd3 * (T.sum(self.W**2)+T.sum(self.W1**2)+T.sum(self.V**2)+T.sum(self.U**2))
        [dw1,du,dv,dw]=T.grad(f_loss, [W1,U,V,W])
        self.sgd_step = theano.function(inputs=[x],
                                        outputs=[f_loss ],
                                        updates=[(self.U, self.U - self.learning_rate * du),
                                                 (self.V, self.V - self.learning_rate * dv),
                                                 (self.W , self.W  - self.learning_rate * dw),
                                                 (self.W1, self.W1 - self.learning_rate * dw1)])

    def predict(self,x):
        U,V,W,W1 = self.U.get_value(),self.V.get_value(),self.W.get_value(),self.W1.get_value()
        def forward_prop_step(x_t, s_t_prev, U, V):
            s_t = np.tanh(np.dot(x_t,U)+ np.dot(s_t_prev,W))
            h3 = np.tanh(np.dot(s_t,V))
            mx=np.argmax(h3,axis=1)
            return [s_t,mx]
        def forward_first_step(x_t, U, V):
            s_t = np.tanh(np.dot(x_t,U))
            h3 = np.tanh(np.dot(s_t,V))
            mx = np.argmax(h3,axis=1)
            return [s_t,mx]
        ans = []
        if len(x.shape) == 1:
            x = np.array([x])
        h1 = np.dot(x,W1)
        last_s,t = forward_first_step(h1,U,V)
        ans.append(t)
        for i in xrange(self.L-1):
            last_s,t = forward_prop_step(h1,last_s,U,V)
            ans.append(t)
        return np.array(ans)


