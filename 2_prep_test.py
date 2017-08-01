import quandl
import os.path
import Tkinter
import numpy as np
import pandas as pd
from sklearn import preprocessing

#-----------------------------------------------------------------------------------------
quandl.ApiConfig.api_key = "INSERT KEY HERE"
# Specify Quandl key to obtain fundamental financial data for testing purposes.
FABRIC_FILE = 'fabric_newtest.csv'
#fabric file is an automatically generated list of the parameter configuration 
#for each Run. This allows user to experiment with different configurations 
#for different datasets and find the optimal solution.
#-----------------------------------------------------------------------------------------

def File_Dialog(file_name):
# Opens a tkinter GUI that allows user to find the necessary documents in other directories if the default file sets specified above are not found.
     
    root = Tkinter.Tk()
    root.withdraw()

    print "The "+file_name+" data file was not found or is not properly formatted."
    print "Please locate a valid file manually."
    return tkFileDialog.askopenfilename(defaultextension="csv",parent=root,title = "Please select "+file_name+" file:")
    

def Get_Factory_Data():
# Opens default files specified above (or prompts user to locate files) and 
#returns the data frames within those files after some pre-processing 
#(ie. parsing dates, filtering out industries than companies, and replacing "
# " with "_" in data sets containing phrases.) 
 
    if os.path.isfile(FABRIC_FILE):
        dat_setup = pd.read_csv(FABRIC_FILE, parse_dates = [1,2])
    else:
        dat_setup = File_Dialog("machine setup data")
    
    par_list = dat_setup.D_PARAM_LIST[3].split('|')
    inc_check = par_list[0]
    del par_list[0]

    if os.path.isfile('firms.csv'):
        dat_meta = pd.read_csv('firms.csv', parse_dates = [5])
    else:
        dat_setu = File_Dialog("firm meta data")
    
    dat_meta.Industry = map(lambda x: str.replace(x,' ','_'), dat_meta.Industry)
        
    if os.path.isfile('former.csv'):

    else:
        dat = File_dialog("fundamental finance data")
    
    if inc_check == "inc":
        dat = dat.filter(items=par_list)
    elif inc_check == "exc":
        dat = dat.drop(par_list,1)


    return [dat_setup, dat_meta, dat]



####################MAIN###################GAME##############


dat_setup, dat_meta, dat = Get_Factory_Data()
n = 0
market = "NASDAQ"
n_seq_size = dat_setup.N_SEQ_SIZE[n]

for n in range(0,len(dat_setup)):
    n_seq_size = dat_setup.N_SEQ_SIZE[n]

    if os.path.isfile("final_lvl"+str(n)+".csv") == False:
        test_frame = pd.DataFrame(columns = list(dat))

        for tkr in list(dat_meta.loc[dat_meta["Exchange"] == market]["Ticker"]):
            temp = quandl.get_table("SHARADAR/SF1", qopts = {'columns': list(dat)}, ticker=tkr, dimension = "MRQ")
            #print temp	
            if len(temp) > n_seq_size:	
                #print tkr

                tempo = temp.fillna(method = 'ffill')
                tempo = tempo.fillna(0)

                test_frame = test_frame.append(temp.loc[len(temp)-n_seq_size:len(temp)-1], ignore_index=True)
                #break
        #print test_frame

        dattst = test_frame.fillna(0)
        dattst["datekey"] = pd.to_numeric(dattst["datekey"])
        dattst["ticker"].to_csv("faust"+str(n)+".csv",index = False, index_label = False)        
        dattst["ticker"] = np. zeros(len(dattst))

        n_in_seq_list = []
        seq_num_list = []
        for i in range(0, int(len(dattst)/n_seq_size)):
            n_in_seq_list.extend(list(range(0,n_seq_size)))
            seq_num_list.extend(np.full(n_seq_size,i))

        dattst['seq_num'] = seq_num_list   
        dattst['n_in_seq'] = n_in_seq_list

        ####copy pasta cant solve it fuck#################
        x = dattst.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        test_frame = pd.DataFrame(x_scaled, columns = list(dattst))
        ##################################################

        test_frame.to_csv("final_lvl"+str(n)+".csv", index = False, index_label = False)




