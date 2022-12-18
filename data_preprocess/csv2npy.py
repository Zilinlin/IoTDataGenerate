import numpy as np
import pandas as pd
import sklearn
import os


cat_cols = ['ip.src','ip.dst','tcp.srcport','tcp.dstport','ip.proto','tcp.flags','tcp.checksum']
num_cols = ['frame.time_relative','frame.len','ip.ttl','tcp.time_delta','tcp.len','tcp.ack','tcp.hdr_len','tcp.window_size_value']
# the numerical label matching is "0,1,2,3"
label = ['benign','scan','infection','flood']


def csv2npy(file_name):
    cat_data = pd.read_csv(file_name, usecols=cat_cols)
    num_data = pd.read_csv(file_name, usecols=num_cols)

    full_cols = cat_cols + num_cols
    print(full_cols)

    full_data = pd.read_csv(file_name, usecols=full_cols)
    #we need to get the data in full_cols order
    full_data = full_data[full_cols]

    #zero the missing data
    num_data = num_data.fillna(value = 0)
    cat_data = cat_data.values
    num_data = num_data.values

    full_data = full_data.values
    # count means the total number of rows
    count = len(full_data)

    #print(full_data)
    #print(num_data)

    # processing the numerical data
    # zero the missing data
    #num_data = num_data.fillna(value=0)

    benign_index = file_name.find("benign")
    scan_index = file_name.find("scan")
    flood_index = file_name.find("flood")

    if benign_index >= 0:
        label = int(0)
    elif scan_index >=0 :
        label = int(1)
    elif flood_index >=0:
        label = int(3)
    else:
        label = int(2)

    label_data = np.full(count, label)
    label_data = label_data.reshape(label_data.shape[0],)
    #print(label_data)

    print(full_data.shape, label_data.shape)

    return full_data, label_data

def main_process(dir_name):

    result_full_data = np.empty((0,15))
    result_label_data = np.empty((0,))

    #print("numpy data", result_cat_data, result_cat_data.shape)
    if os.path.isdir(dir_name):
        for file_name in os.listdir(dir_name):
            #print("file name: ",file_name)

            full_file_name = dir_name + file_name
            part_full_data, part_label_data = csv2npy(full_file_name)

            result_full_data = np.concatenate((result_full_data,part_full_data),axis=0)
            result_label_data = np.concatenate((result_label_data, part_label_data),axis=0)

    print(result_full_data.shape)
    print(result_label_data.shape)

    np.save("data.npy",result_full_data)
    np.save("label.npy",result_label_data)

    return 0

# this is for not processing all the csv data
def process_part(dir_name):
    result_full_data = np.empty((0,15))
    result_label_data = np.empty((0,))

    process_file_list = ['benign-dec.pcap_.csv','mirai-ackflooding-2-dec.pcap_.csv','mirai-hostbruteforce-3-dec.pcap_.csv',
                         'scan-hostport-3-dec.pcap_.csv']

    for file_name in process_file_list:
        full_file_name = dir_name + file_name
        temp_full_data, temp_label_data = csv2npy(full_file_name)

        result_full_data = np.concatenate((result_full_data, temp_full_data), axis=0)
        result_label_data = np.concatenate((result_label_data, temp_label_data), axis = 0)

    print(result_full_data.shape)
    print(result_label_data.shape)

    # transfer pd array to numpy array
    # data = np.array(result_full_data)
    # label = np.array(result_label_data)

    np.save("data.npy", result_full_data)
    np.save("label.npy", result_label_data)

    return 0


# delete all the rows that have NaN
def drop_nan_rows():
    data = np.load('data.npy', allow_pickle = True)
    label = np.load('label.npy', allow_pickle = True)

    #if_nan = ~np.isnan(data).any(axis=1)
    if_nan = ~pd.isna(data).any(axis=1)

    result_data = data[if_nan,:]
    result_label = label[if_nan]

    print(result_data.shape)
    print(result_label.shape)

    data = np.array(result_data)
    label = np.array(result_label)

    label = label.astype(int)

    np.save('data.npy',data)
    np.save('label.npy',label)

    return 0


# split the data to training, testing and validating data
def data_split():
    data = np.load('data.npy',allow_pickle=True)
    label = np.load('label.npy',allow_pickle=True)

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.2,random_state=0,stratify=label)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.125,random_state=0,stratify=y_train)

    print("training data shape:", x_train.shape)
    print("training label shape:", y_train.shape)
    print("validating data shape:", x_val.shape)
    print("validating label shape:", y_val.shape)
    print("testing data shape:", x_test.shape)
    print("testing label shape:", y_test.shape)

    X_cat_train = x_train[:,0:7]
    X_num_train = x_train[:,7:]
    X_cat_test = x_test[:,0:7]
    X_num_test = x_test[:,7:]
    X_cat_val = x_val[:,0:7]
    X_num_val = x_val[:,7:]

    X_num_train = np.nan_to_num(X_num_train)
    X_num_test = np.nan_to_num(X_num_test)
    X_num_val = np.nan_to_num(X_num_val)

    np.save('iot_intrusion/X_cat_train.npy',X_cat_train.astype(object))
    np.save('iot_intrusion/X_num_train.npy',X_num_train.tolist())
    np.save('iot_intrusion/X_cat_test.npy',X_cat_test.astype(object))
    np.save('iot_intrusion/X_num_test.npy',X_num_test.tolist())
    np.save('iot_intrusion/X_cat_val.npy',X_cat_val.astype(object))
    np.save('iot_intrusion/X_num_val.npy',X_num_val.tolist())
    np.save('iot_intrusion/y_train.npy',y_train.tolist())
    np.save('iot_intrusion/y_test.npy',y_test.tolist())
    np.save('iot_intrusion/y_val.npy',y_val.tolist())






def csv2npy_old(file_name):


    # clean the panda data
    data = data.dropna(axis=1, how='all')
    #print(data)
    #print(data.keys())

    print(data['ip.src'])
    # zero the missing value
    data = data.fillna(value=0)
    data_ = data.to_numpy()

    #standard scaler
    from sklearn.preprocessing import StandardScaler
    data_ = StandardScaler().fit_transform(data_)
    print(type(data_))
    print(data_)

#csv2npy('../csv_iot_intrusion_dataset/scan-hostport-1-dec.pcap_.csv')
#csv2npy('../csv_iot_intrusion_dataset/benign-dec.pcap_.csv')
#process_part('../csv_iot_intrusion_dataset/')
drop_nan_rows()
data_split()
