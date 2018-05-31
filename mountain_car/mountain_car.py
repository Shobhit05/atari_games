import gym
import tensorflow as tf
import numpy as np
import random
import math
import time

env = gym.make("MountainCar-v0")


# 0 for left 1 for staying 2 for going right
inpt=tf.placeholder(dtype=tf.float32,shape=[None,2])
target=tf.placeholder(dtype=tf.float32,shape=[None,3])

w1=tf.Variable(tf.random_normal(shape=[2,64],dtype=tf.float32))
b1=tf.Variable(tf.random_normal(shape=[64],dtype=tf.float32))

w2=tf.Variable(tf.random_normal(shape=[64,32],dtype=tf.float32))
b2=tf.Variable(tf.random_normal(shape=[32],dtype=tf.float32))

w3=tf.Variable(tf.random_normal(shape=[32,3],dtype=tf.float32))
b3=tf.Variable(tf.random_normal(shape=[3],dtype=tf.float32))

layer1=tf.add(tf.matmul(inpt,w1),b1)
layer2=tf.add(tf.matmul(layer1,w2),b2)
output=tf.add(tf.matmul(layer2,w3),b3)

loss=tf.reduce_mean(tf.square(target-output))
trainer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

replay_memory=[]
min_samples=5000
batch_size=100



with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    i=0
    while True:
        i+=1
        count=0
        tot_reward=0
        done=False
        state=env.reset()
        while done==False:
            count+=1
            epsilon = 0.1 + (1.0 - 0.1)*(math.exp(-0.001*i))
            neural_net_output=sess.run(output,feed_dict={inpt:[state]})[0]

            action=np.argmax(neural_net_output)
            
            if random.random() < epsilon:
                action=random.randint(0,1)
                if action==1:
                    action=2
                    
            next_state,reward,done,info=env.step(action)
            reward=min(state[0],0.5)+0.5
            tot_reward+=reward

            next_batch_out=sess.run(output,feed_dict={inpt:[next_state]})[0]

            max_q=np.max(next_batch_out)
            
            target_batch=neural_net_output[:]
            
            target_batch[action]=reward+0.99*max_q

            replay_memory.append([state,target_batch])

            sample_size=min(batch_size,len(replay_memory))

            random_batch=random.sample(replay_memory,sample_size)

            inpt_batch=[d[0] for d in random_batch]

            target_batch=[d[1] for d in random_batch]
            
                        
            sess.run(trainer,feed_dict={inpt:inpt_batch,target:target_batch})
            
            env.render()
        
            state=next_state
            count+=1
            
            if len(replay_memory)>50000:
                replay_memory.pop(0)

        print "Reward in %s step %s epsilon value %s" %(i,tot_reward,epsilon)

            
                
            


        














        
        
