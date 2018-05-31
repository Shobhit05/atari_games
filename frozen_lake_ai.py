import gym
import tensorflow as tf
import numpy as np
import time
import os

env = gym.make("FrozenLake-v0").env

def do_one_hot_encoding(state):
    arr=np.zeros([1,16])
    arr[0][state]=1.0
    return arr
    
# left->0,down->1,right->2,up->3
# wind is there so it is impossible that agent goes to intended next states

inpt=tf.placeholder(tf.float32,shape=[1,16])
aout=tf.placeholder(tf.float32,shape=[1,4])

weights=tf.Variable(tf.random_normal([16,4],dtype=tf.float32))
bias=tf.Variable(tf.random_normal([4],dtype=tf.float32))

output=tf.matmul(inpt,weights)

action=tf.argmax(output,1)[0]
####   aout=== actual output ##### 
cost=tf.reduce_sum(tf.square(aout-output))

trainer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
e = 0.1
win=0
episode=0

while episode<1:
    episode+=1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,"froz_mod.ckpt")
        for i in range(0,2000):
            current_state=env.reset()
            j=0
            while j<99:
                j+=1
                action_taken,q_values=sess.run([action,output],feed_dict={inpt:do_one_hot_encoding(current_state)})
                if np.random.rand(1) < e:
                        action_taken = env.action_space.sample()

                next_state,reward,done,info=env.step(action_taken)

                next_state_q_values=sess.run(output,feed_dict={inpt:do_one_hot_encoding(next_state)})

                max_q_value=np.max(next_state_q_values)

                target_q_values=q_values
                
                target_q_values[0,action_taken]=reward+0.99*(max_q_value)

                loss=sess.run([trainer],feed_dict={aout:target_q_values,inpt:do_one_hot_encoding(current_state)})

                current_state=next_state
                #env.render()
                if reward==1:
                    win+=1
                    #env.render()
                    break
                    
                if done==True:
                    e = 1./((i/50) + 10)
                    break
        saver.save(sess, '/home/shobhit/Desktop/Neural Network/atari_games/froz_mod.ckpt')


    print "The Game won by machine"+str(win)      


    
            



