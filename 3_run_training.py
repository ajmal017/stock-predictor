

import tensorflow as tf
import tensorflow.contrib as tfc
import pandas as pd
import numpy as np
import Tkinter
import tkFileDialog
import os.path

#-----------------------------------------------------------------------------------------
TEMP_DIR = 7

FABRIC_FILE = "fabric_newtest"
#fabric file is an automatically generated list of the parameter configuration 
#for each Run. This allows user to experiment with different configurations 
#for different datasets and find the optimal solution.
#-----------------------------------------------------------------------------------------
def File_Dialog(file_name):
	
	root = Tkinter.Tk()
	root.withdraw()

	print "The "+file_name+" data file was not found or is not properly formatted."
	print "Please locate a valid file manually."
	return tkFileDialog.askopenfilename(defaultextension="csv",parent=root,title = "Please select "+file_name+" file:")
	

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
def Get_Foundation_Data(file_name, n = None):
	
	if n == None:
		file_name = file_name+".csv"
	else:
		file_name = file_name+str(n)+".csv"
		
	if os.path.isfile(file_name):
		dat = pd.read_csv(file_name, infer_datetime_format = True)
	else:
		dat = pd.read_csv(File_Dialog(file_name), infer_datetime_format = True)
	
	return dat


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
def Split_Dataset(dat,dat_setup,n):
	n_dtpnts = len(dat)
	n_seqs = len(dat)/dat_setup.N_SEQ_SIZE[n]
	
	n_dtpnts_train = int (n_seqs * dat_setup.N_TRAIN_RATIO[n])*dat_setup.N_SEQ_SIZE[n]
	n_dtpnts_test = int (n_seqs * dat_setup.N_TEST_RATIO[n])*dat_setup.N_SEQ_SIZE[n]
	n_dtpnts_predict = int (n_seqs * dat_setup.N_PREDICT_RATIO[n])*dat_setup.N_SEQ_SIZE[n]
	
	dat_train = dat.loc[:n_dtpnts_train-1]
	dat_test = dat.loc[n_dtpnts_train:n_dtpnts_train+n_dtpnts_test-1]
	dat_predict = dat.loc[n_dtpnts_train+n_dtpnts_test:n_dtpnts_train+n_dtpnts_test+n_dtpnts_predict-1]
	
	return dat_train, dat_test, dat_predict


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
def Define_Data_and_Result(dat, mset, n):
	
	n_seqs = int(len(dat)/mset.N_SEQ_SIZE[n])
	#print mset.N_SEQ_SIZE[n]
	mask_fin = dat.index%mset.N_SEQ_SIZE[n] < (mset.N_SEQ_SIZE[n] -1)
	mask_curr = dat.index%mset.N_SEQ_SIZE[n] == 0
	
	data = dat.loc[mask_fin]
	cur = dat.loc[mask_curr].filter(items=["price"]).set_index([list(range(0,n_seqs))])
	fin = dat.loc[~mask_fin].filter(items=["price"]).set_index([list(range(0,n_seqs))])
	
	res = pd.concat([cur, fin], axis = 1, ignore_index = True)
	
	res["res"] = np.where(res[1]>res[0], 1, 0)
	res["ser"] = np.where(res[1]>res[0], 0, 1)
	result = res.filter(["res","ser"])
	
	return data, result

#-----------------------------------------------------------------------------------------
def RunSesh(mset,n):

	dat = Get_Foundation_Data("forged",n)
	dat_train, dat_test, dat_predict = Split_Dataset(dat, mset, n)
	
	data_train, res_train = Define_Data_and_Result(dat_train, mset, n)
	data_test, res_test = Define_Data_and_Result(dat_test, mset, n)
	data_predict, res_predict = Define_Data_and_Result(dat_predict, mset, n)
	
	dat_truetest = Get_Foundation_Data("final_lvl",n)
	data_truetest, res_truetest = Define_Data_and_Result(dat_truetest, mset, n)

	N_STEP = int(mset.N_BATCH_SIZE[n])*int(mset.N_SEQ_SIZE[n]-1)
	N_BATCHES = int(len(data_train)/N_STEP)
#-----------------------------------------------------------------------------------------
#TF Setup---------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
	tf.reset_default_graph()
#-----------------------------------------------------------------------------------------
	with tf.name_scope('Input'):
		with tf.name_scope('InputData'):
			data_list = tf.placeholder(tf.float32,[None,int(data_train.shape[1])], name = "DataList")
		with tf.name_scope('Reshape'):
			data_struct = tf.reshape(data_list, [-1,int(mset.N_SEQ_SIZE[n]-1),int(data_train.shape[1])])

	with tf.name_scope('Result'):
		target = tf.placeholder(tf.float32,[None,2], name = "Target")

#-----------------------------------------------------------------------------------------
	
	with tf.name_scope('LSTMlayer'):
		cell = tfc.rnn.BasicLSTMCell(mset.N_HIDDEN_LAYERS[n])

	with tf.name_scope('RNNlayer'):
		with tf.name_scope('DynamicRNN'):
			val, state = tf.nn.dynamic_rnn(cell, data_struct, dtype = tf.float32)
		with tf.name_scope('Transponse'):
			vale = tf.transpose(val,[1,0,2])
		with tf.name_scope('Gather'):
			last = tf.gather(vale, int(vale.get_shape()[0]-1))

#-----------------------------------------------------------------------------------------

	weight_s1 = tf.Variable(tf.random_normal([mset.N_HIDDEN_LAYERS[n],int(target.get_shape()[1])]), name = "W")
	bias_s1 = tf.Variable(tf.constant(1.0,shape = [target.get_shape()[1]]), name = "B")

#-----------------------------------------------------------------------------------------

	with tf.name_scope('OutputLayer'):
		prediction = tf.nn.softmax(tf.matmul(last,weight_s1)+bias_s1)


	with tf.name_scope('CrossEntropy'):
		cross_entropy = -tf.reduce_sum(target*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
	
	with tf.name_scope('Optimizer'):
		with tf.name_scope('Adam'):
			optimizer  = tf.train.AdamOptimizer()
		with tf.name_scope('Minimize'):
			minimize = optimizer.minimize(cross_entropy)

	with tf.name_scope('Accuracy'):
		with tf.name_scope('NotEqual'):
			mistakes = tf.not_equal(tf.argmax(target,1), tf.argmax(prediction,1))
		with tf.name_scope('ReduceMean'):
			error = tf.reduce_mean(tf.cast(mistakes,tf.float32))


######LLLLLEEEEEETTTTS####GEEETTTTT#####REEEAAAAAAAAAAAAAAAAAADDDTTTTT######TOOOOOO#######################################################################################
############RRRRUUUUUMMMMMMMMMMMMMMMMMMMMMMMBBBBBBBBBBBBBBBBBBBBBLLLLLLLLLLLLEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

	init_op = tf.global_variables_initializer()

# Tensorboard summaries-------------------------------------------------------------------
	tf.summary.scalar("CrossEnt",cross_entropy)
	#tf.summary.scalar("Mistakes", mistakes)
	tf.summary.scalar("Error", error)
	#tf.summary.scalar("Target", target) 

	tf.summary.histogram("Weights", weight_s1)
	tf.summary.histogram("Biases", bias_s1)
	tf.summary.histogram("Activation", prediction)
#-----------------------------------------------------------------------------------------

	merged_summary = tf.summary.merge_all()
	
	with tf.Session() as sess:
		sess.run(init_op)

# Tensorboard filewriter------------------------------------------------------------------
		writer = tf.summary.FileWriter("/home/sunspot/Desktop/rebuilt/temp/"+str(TEMP_DIR))
#-----------------------------------------------------------------------------------------
		best = 100
		for i in range(mset.N_TOTAL_EPOCHS[n]):
			pntr = 0
			while pntr < N_BATCHES: 

				#skip forward one data set if random number is less than dropout rate 
				if np.random.choice(10,1) < mset.N_DROPOUT_RATE[n]*10:
					pntr += 1

				#define training data set and result vector
				inp = data_train[pntr*N_STEP:(pntr+1)*N_STEP]
				outp = res_train[pntr*mset.N_BATCH_SIZE[n]:(pntr+1)*mset.N_BATCH_SIZE[n]]
				_, summary = sess.run([minimize, merged_summary],{data_list:inp, target: outp})
				
				pntr += 1 
			
			incorrect = sess.run(error,{data_list:data_test, target:res_test})
			
			if incorrect < best:
				best = incorrect
				best_i = i
		

		print(('Epoch {:2d} accuracy {:3.10f}%'.format(best_i+1,100-100*best)))
		
		writer.add_graph(sess.graph)
		writer.add_summary(summary,best_i)

		sess.close()


#MAIN#####################################################################################
mset = Get_Foundation_Data(FABRIC_FILE)
n = len(mset)-1

for n in range(len(mset)):
	print ("Run - " + str(n))	
	RunSesh(mset, n)
##########################################################################################






