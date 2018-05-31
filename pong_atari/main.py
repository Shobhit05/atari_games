import gym
import time
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import random
import numpy as np
env = gym.make('BreakoutDeterministic-v4')

def get_resizable_frame(frame):
    frame[0:16,0:160]=255
    frame = cv2.cvtColor(cv2.resize(frame, (84,84)), cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)
    
    ############## Crop the Image if needed ############################
    #print frame.shape
    #plt.imshow(frame,'gray')
    #plt.show()
    
    return frame


inpt=tf.placeholder(dtype=tf.float32,shape=[None,84,84,4])

conv1={"weight":tf.Variable(tf.zeros([8,8,4,32],dtype=tf.float32)),
       "bias":  tf.Variable(tf.zeros([32],dtype=tf.float32))
       }

conv2={"weight":tf.Variable(tf.zeros([4,4,32,64],dtype=tf.float32)),
       "bias":  tf.Variable(tf.zeros([64],dtype=tf.float32))
       }

conv3={"weight":tf.Variable(tf.zeros([3,3,64,64],dtype=tf.float32)),
       "bias":  tf.Variable(tf.zeros([64],dtype=tf.float32))
       }

fcl1={"weight":tf.Variable(tf.zeros([3136,784],dtype=tf.float32)),
      "bias":  tf.Variable(tf.zeros([784],dtype=tf.float32))
      }

fcl2={"weight":tf.Variable(tf.zeros([784,4])),
      "bias":tf.Variable(tf.zeros([4]))
              }

outconv1=tf.nn.relu(tf.nn.conv2d(inpt,conv1["weight"],strides=[1,4,4,1],padding="VALID")+
                 conv1["bias"])

outconv2=tf.nn.relu(tf.nn.conv2d(outconv1,conv2["weight"],strides=[1,2,2,1],padding="VALID")+
                 conv2["bias"])

outconv3=tf.nn.relu(tf.nn.conv2d(outconv2,conv3["weight"],strides=[1,1,1,1],padding="VALID")+
                 conv3["bias"])

outconv3=tf.reshape(outconv3,[-1,3136])

fc1=tf.nn.relu(tf.matmul(outconv3,fcl1["weight"])+fcl1["bias"])

output=tf.matmul(fc1,fcl2["weight"])+fcl2["bias"]


action_mat = tf.placeholder(dtype=tf.float32,shape=[None,4]) 

aout = tf.placeholder(dtype=tf.float32,shape=[None]) 

action = tf.reduce_sum(tf.multiply(output, action_mat), reduction_indices = 1)

cost = tf.reduce_mean(tf.square(action - aout))

trainer = tf.train.AdamOptimizer(1e-6).minimize(cost)


replay_memory=[]
batch_size=100
min_samples=50000
e=0.30

curr_frame=get_resizable_frame(env.reset())

frame_arr=np.stack((curr_frame,curr_frame,curr_frame,curr_frame),axis=2)


count=0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    try:
        saver.restore(sess,"pong1.ckpt")
        print "Restored"
        time.sleep(1)
    except:
        pass

    while True:

        count+=1
        
        out_l=sess.run([output],feed_dict={inpt:[frame_arr]})[0]
        
        action_choosen=np.argmax(out_l)
        

        action_matrix=np.zeros([4])
        
        if(random.random() <= e):
            print "Random Action",
            action_choosen=env.action_space.sample()
        else:
            print "Neural Network Predict",
        
        if e >0.05:
            e -= (1.0 - 0.05) / 500000

        action_matrix[action_choosen]=1

        next_frame,reward,done,info=env.step(action_choosen)

        next_frame=get_resizable_frame(next_frame)
        
        next_frame=np.reshape(next_frame,(84,84,1))

        next_frame=np.append(next_frame,frame_arr[:, :, 0:3], axis = 2)
        
        
        replay_memory.append([frame_arr,action_matrix,reward,next_frame])

        if len(replay_memory)>500000:
            replay_memory.pop(0)
            
        if count > 50000:
            random_batch=random.sample(replay_memory,batch_size)
            inp_batch,action_mat_batch,reward_batch,next_frame_batch=[],[],[],[]
            for batch in random_batch:
                inp_batch.append(batch[0])
                action_mat_batch.append(batch[1])
                reward_batch.append(batch[2])
                next_frame_batch.append(batch[3])
                
            
            actual_output=[]
            out_val=sess.run([output],feed_dict={inpt:next_frame_batch})[0]
         
            for i in range(len(random_batch)):
                actual_output.append(reward_batch[i]+0.99*np.max(out_val[i]))
            
            sess.run([trainer],feed_dict={inpt:inp_batch,
                                          aout:actual_output,
                                          action_mat:action_mat_batch
                                          })

        frame_arr=next_frame

        if done:
            curr_frame=get_resizable_frame(env.reset())
            frame_arr=np.stack((curr_frame,curr_frame,curr_frame,curr_frame),axis=2)
            #saver.save(sess,"/home/shobhit/Desktop/Neural Network/atari_games/pong_atari/pong.ckpt")

        if count%10000==0:
            saver.save(sess,"/home/shobhit/Desktop/Neural Network/atari_games/pong_atari/pong1.ckpt")


        print("TIMESTEP-", count, "EPSILON-", e, "ACTION",action_choosen,"REWARD-",reward,"max",np.max(out_l))

        #print env.unwrapped.get_action_meanings()
        
     

        env.render()
        #time.sleep(0.05)



