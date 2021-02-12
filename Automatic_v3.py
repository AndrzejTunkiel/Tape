# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:35:54 2021

@author: lloth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import MinMaxScaler

#%%
df = pd.read_csv('f9ad.csv')

def optimize_me(hGRU,
                hAttrCount,
                hDense1,
                hMemoryMeters,
                hstep_extension,
                hConv1,
                hConv1k,
                hPoolSize,
                hDense2):

    
    hyperlist = [hGRU,
                hAttrCount,
                hDense1,
                hMemoryMeters,
                hstep_extension,
                hConv1,
                hConv1k,
                hPoolSize,
                hDense2]
    
    for i in range(len(hyperlist)):
        hyperlist[i] = int(hyperlist[i])
    
    [hGRU,
    hAttrCount,
    hDense1,
    hMemoryMeters,
    hstep_extension,
    hConv1,
    hConv1k,
    hPoolSize,
    hDense2]   = hyperlist 
    
    
    imagination_meters = 25
    
    #step_extension = 1 
    #min_required_variance = 0.95
    
    df = pd.read_csv('f9ad.csv')
    
    ## List attributes
    # print('Listing all the attributes in the loaded dataset:')
    # print(list(df))
    
    #%%
    
    ## Drop unwanted columns
    
    drops = ['Unnamed: 0', 'Unnamed: 0.1', 'RHX_RT unitless', 'Pass Name unitless',
             'nameWellbore', 'name','RGX_RT unitless',
                       'MWD Continuous Azimuth dega']
    
    df = df.drop(columns = drops)
    df = df[1900:]   ## Done this to include more data. Smart selection needed!
    #%%
    
    ## Select target attribute and index.
    
    index = 'Measured Depth m'
    target =  'MWD Continuous Inclination dega'
    
    
    #%%
    
    ## Get basic statistics extraction
    #
    # s = global statistics
    # m = matrix of gap lengths
    # per = statistics per attribute
    
    from statistics_module import stats
    
    s, m, per = stats(df)
    
    #%%
    
    ## Gap statistics for target
    #
    # This chart will show the percentage of dataset occupied by gaps of a certain
    # size. Gaps are normal in drilling logs and nothing to be afraid of
    
    x_label = per[target]['gap_sizes']
    x = np.arange(0, len(x_label),1)
    
    y = per[target]['percentage_cells_occupied'] 
    # plt.xticks(x, x_label, rotation=90)
    # plt.bar(x,y)
    # plt.title(f'Gap distribution in:\n {target}')
    # plt.xlabel('Gap length')
    # plt.ylabel('Percentage of dataset occupied')
    
    x_labels = x.tolist()
    x_labels[0] = 'data'
    # plt.xticks(x, x_labels)
    # plt.grid()
    # plt.show() 
    
    #%%
    
    ## Outlier detection
    
    outlier_cutoff = 0.005 #arbitrarily selected
    
    # calculation that penalizes long, rare, continuous gaps
    out_coef = per[target]['gap_sizes'] / (per[target]['gap_counts'] * len(df))
    
    x = np.arange(0,len(per[target]['gap_sizes']),1)
    x_label = per[target]['gap_sizes']
    x = np.arange(0, len(x_label),1)
    
    # plt.xticks(x, x_label, rotation=90)
    # plt.bar(x,out_coef)
    # #plt.ylim(0,0.005)
    # plt.plot(x,[outlier_cutoff]*len(x), color='red', label='cutoff')
    # plt.legend()
    x_labels = x.tolist()
    x_labels[0] = 'data'
    # plt.xticks(x, x_labels)
    # plt.title(f'Gap occupancy = gap size / (relative gap quantity) \n in:{target}')
    # plt.xlabel('Gap length')
    # plt.ylabel('Gap occupancy')
    # plt.grid()
    # plt.show() 
    
    #%%
    
    ## Automatic proposal of useful area, part 1
    
    # find the smallest outlier gap
    # outlier coefficients - True when above outlier cutoff
    cutoff_map = (out_coef >= outlier_cutoff)
    
    # Using map created above, what is the smallest outlier?
    lower_cutoff = np.min(np.asarray(per[target]['gap_sizes'])[cutoff_map])
    
    # This is done to quickly mark gaps bigger than cutoff with zero and
    # other with NaN. This makes a good chart.
    from functools import partial
    
    def _cutoff(x, lower_cutoff=0):
        if x >= lower_cutoff:
            return 0
        else:
            return np.nan
    
    _cutoff_par = partial(_cutoff, lower_cutoff=lower_cutoff)
    mapped_outliers = list(map(_cutoff_par, m[target]))
    
    # plt.scatter(df[index], df[target], s=1, c='black', label='data')
    # plt.plot(df[index], mapped_outliers, c='red', label='Unusuable range') 
    #                                                 # has to be plot to avoid index
    #                                                 # discontinuities
                                                    
    # plt.grid()
    # plt.legend()
    # plt.title('Useful range analysis')
    # plt.xlabel(f'{index}')
    # plt.ylabel(f'{target}')
    # plt.show()
    #%%
    
    ## Automatic proposal of useful area, part 2
    
    # Simply finds the biggest area with acceptable gaps
    
    # TODO - check if the algorithm will detect a stride at the end of the dataset
    #        because I have a feeling it won't!
    
    strides = []
    
    s_start = -1
    s_stop = -1
    
    for i in range(len(df)):
        if mapped_outliers[i] != 0 and s_start == -1:
            s_start = i
        elif mapped_outliers[i] == 0 and s_start != -1:
            s_stop = i
            strides.append([s_start, s_stop, s_stop - s_start ])
            
            s_start = -1
            s_stop = -1
    
    strides = np.asarray(strides)
    strides = strides[strides[:,2].argsort()][::-1] # sort by length [2] 
                                                    # and reverse
    # print(f'''Proposed range to use is row {strides[0,0]} to row {strides[0,1]}
    # for total of {strides[0,0]} rows
    #       ''')
    # print(f'All found strides are: [start, stop, length]')
    # print(strides)
    
    s_start = strides[0,0]
    s_stop = strides[0,1]
    #%%
    
    ## cut the dataframe for the selected stride, redo the stats
    #  From now on dfs is used (dataframe stride)
    margin_percent = 1  # since the edges can be a bit unpredictable, margin is
                        # removed
    
    s_start = s_start + int((s_stop - s_start) * 0.01*margin_percent)
    s_stop = s_stop - int((s_stop - s_start) * 0.01*margin_percent)
    dfs = df.iloc[s_start:s_stop] # dfs = DataFrameStride
    s, m, per = stats(dfs)
    
    #%%
    
    ## Removing columns that contain big gaps
    
    from functools import partial
    
    def _cutoff_inv(x, lower_cutoff=0):
        if x >= lower_cutoff:
            return 1
        else:
            return 0
    
    _cutoff_par = partial(_cutoff_inv, lower_cutoff=lower_cutoff)
    
    killed_cols = []
    
    for column in list(dfs):
        mapped_outliers = list(map(_cutoff_par, m[column]))
        offender_count = np.sum(mapped_outliers)
        if offender_count > 0:
            dfs = dfs.drop(columns = [column])
            killed_cols.append([column, 100*offender_count/len(dfs)])
    
    killed_cols = pd.DataFrame(killed_cols, columns=['Name','Percent offending'])
    #print('Removed following columns due to outlier gap (showing under 15% only):')
    #print(killed_cols.sort_values(by='Percent offending')[killed_cols['Percent offending'] < 15])
    
    #%%
    
    ## Checking if first derivative of index is stable.
    
    index_dr = np.diff(dfs[index])
    
    index_mean = np.mean(index_dr)
    index_std = np.std(index_dr)
    index_maxgap = np.max(index_dr)
    deviation = np.abs(index_dr - index_mean)/index_std
    
    #print(f'Maximum distance from mean is {np.max(deviation):.1f} standard deviations')
    #print(f'If this value is above 6, there may be too high sampling frequency variation')
    
    #%%
    
    ## Counting zeros in the first derivative to see if it should be ffilled
    ## or linearly interpolated
    
    ## NOTE: Actual filling will not happen here, but AFTER the data split
    
    fill_method = {}
    
    for attribute in list(dfs):
        
        dropna_diff = np.diff(dfs[attribute].dropna())
        zeros_p = np.count_nonzero(dropna_diff == 0) / len(dropna_diff)
        
        if zeros_p > 0.9: # Threshold to check?
            fill_method[attribute] = 'ffill'
        else:
            fill_method[attribute] = 'linterp'
    
    
    #%%
    
    #%% 
    
    ## Gap filling - but only forward filling. Linear interpolation is done later
    
    for attribute in list(dfs):
        if fill_method[attribute] == 'ffill':
            dfs[attribute].ffill(inplace=True)
    
    #%%
    
    
    
    #%%
    
    ## DATA SPLIT
    
    split = 0.65 #portion of data available
    future = 0.2 #section after available, for testing
    
    X = dfs.drop(target, axis=1)
    y = dfs[target].to_frame()
    
    splitpoint = int(len(dfs)*split)
    futurepoint = int(len(dfs)*(split+future))
    
    X_train = X[:splitpoint]
    y_train = y[:splitpoint]
    X_test  = X[splitpoint:futurepoint]
    y_test  = y[splitpoint:futurepoint]
    
    
    #%%
    
    ## Linear interpolation after split, so the future does not leak.
    
    
    for attribute in list(X_train):
        if fill_method[attribute] == 'linterp':
            X_train = X_train.interpolate().ffill().bfill()
            
    for attribute in list(y_train):
        if fill_method[attribute] == 'linterp':
            y_train = y_train.interpolate().ffill().bfill()
            
    for attribute in list(X_test):
        if fill_method[attribute] == 'linterp':
            X_test = X_test.interpolate().ffill().bfill()
            
    for attribute in list(y_test):
        if fill_method[attribute] == 'linterp':
            y_test = y_test.interpolate().ffill().bfill()
    
    
    
    
    
    
    
    #%%
    
    ## Resampling
    
    from sklearn.neighbors import RadiusNeighborsRegressor
    
    reg = RadiusNeighborsRegressor()
    
    step_length = index_mean * hstep_extension
    
    i_train_min = np.min(X_train[index])
    i_train_max = np.max(X_train[index])
    i_test_min = np.min(X_test[index])
    i_test_max = np.max(X_test[index])
    
    index_train = np.arange(i_train_min, i_train_max, step_length).reshape(-1,1)
    index_test = np.arange(i_test_min, i_test_max, step_length).reshape(-1,1)
    
    
    reg = RadiusNeighborsRegressor(radius=index_maxgap, weights='distance')
    
    reg.fit(X_train[index].to_numpy().reshape(-1,1), y_train[target].to_numpy())
    y_train = pd.DataFrame()
    y_train[target] = reg.predict(index_train)
    
    reg.fit(X_test[index].to_numpy().reshape(-1,1), y_test[target].to_numpy())
    y_test = pd.DataFrame()
    y_test[target] = reg.predict(index_test)
    
    X_train_resampled = pd.DataFrame()
    for attribute in list(X_train):
        reg.fit(X_train[index].to_numpy().reshape(-1,1), X_train[attribute].to_numpy())
        X_train_resampled[attribute] = reg.predict(index_train)
        
    X_train = X_train_resampled
    
    
    
    X_test_resampled = pd.DataFrame()
    for attribute in list(X_train):
        reg.fit(X_test[index].to_numpy().reshape(-1,1), X_test[attribute].to_numpy())
        X_test_resampled[attribute] = reg.predict(index_test)
        
    X_test = X_test_resampled
    #%%
    
    ## Inclination to delta inclination convertion needed here!
    
    convert_to_diff = []#['MWD Continuous Inclination dega']
    lcs_list = ['MWD Continuous Inclination dega'] #list of parameters that are to be in local coordinate system
    
    for attr in convert_to_diff:
        if attr == target:
            y_train[attr] = y_train[attr].diff().bfill() #bfill to kill initial NaN
            y_test[attr] = y_test[attr].diff().bfill()
        else:
            X_train[attr] = X_train[attr].diff().bfill()
            X_test[attr] = X_test[attr].diff().bfill()
    #%%
    
    ## Scaling the data. Note that range is decide on the training dataset only!
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train[X_train.columns] = scaler_X.fit_transform(X_train[X_train.columns])
    y_train[y_train.columns] = scaler_y.fit_transform(y_train[y_train.columns])
    
    ## Test portion is tranformed based on the existing scaler
    X_test[X_test.columns] = scaler_X.transform(X_test[X_test.columns])
    y_test[y_test.columns] = scaler_y.transform(y_test[y_test.columns])
    
    
    #%%
    
    ## Dataframe for use in correlation analysis, where X_train and y_train is
    ## together
    df_train = X_train
    df_train = df_train.merge(y_train, how='outer', left_index=True,
                              right_index=True)
    
    #%%
    
    ## Choice of attribute selection method done on the complete dataset
    ## NOTE: PCA has to be applied AFTER the split, never before!
    
    # print('''Choose attribute selection method:
    #     1) pearson coefficient
    #     2) PCA
    #     3) ppscore''')
        
    asel_choice = '2'#input('Your choice:')
    
    #%%
    ## Simple correlation, pearson, straight from Pandas
    
    ## Note that correlations are re-done after each split, and done only on the
    ## training dataset!
    
    ## [] Ensure that Index is carried forward!
    
    PCA_n = -1 #i.e. not in use
    
    
    if asel_choice == '1':
        
        dfs_corr = df_train.corr(method='pearson')
        corr_values = dfs_corr[target].to_numpy()
        corr_index = dfs_corr[target].index.to_numpy()
        
        corr_m = np.column_stack((corr_values, corr_index))
        
        for i in range(len(corr_m)):
            if np.isnan(corr_m[i,0]):
                corr_m[i,0] = 0
            else:
                corr_m[i,0] = np.abs(corr_m[i,0])
        
        corr_m = corr_m[corr_m[:,0].argsort()]
        
        keep_columns = corr_m[-1-hAttrCount:-1,1]
        # cutoff = np.nanpercentile(np.abs(corr_values),50) #correlation cutoff
        
        # keep_columns = []
        # for i in range(len(corr_index)):
        #     if np.abs(corr_values[i]) >= cutoff and corr_index[i] != target:
        #         keep_columns.append(corr_index[i])
    
        X_train = X_train[keep_columns]
        X_test  = X_test[keep_columns]
    
    
    
    ## PCA based 
    
    
    elif asel_choice == '2':
        from sklearn.decomposition import PCA
        
        keep_columns = [] #empty for future code compatibility
        # pca = PCA().fit(X_train)
        
        # explained_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # # plt.plot(explained_variance)
        # # plt.xlabel('number of components')
        # # plt.ylabel('cumulative explained variance');
        # # plt.show()
        
        #  # arbitrary number
        
        # for i in range(len(explained_variance)):
        #     if explained_variance[i] >= min_required_variance:
        #         break
            
        PCA_n = hAttrCount
        
        # applied after sensitivity analysis
    
    ## ppscore based
    
    elif asel_choice == '3':
        import ppscore as pps
        dfs_corr = pps.predictors(df_train, target, output='list')
        
    
        min_required_ppscore = 0.3
        
        keep_columns = []
        for i in range(len(dfs_corr)):
            if dfs_corr[i]['ppscore'] > min_required_ppscore:
                keep_columns.append(dfs_corr[i]['x'])
                
        X_train = X_train[keep_columns]
        X_test  = X_test[keep_columns]
    
        
    else:
        sys.exit("Error, incorrect attribute selection choice")
    
    
    #%%
    
    ## Sensitivity study gets applied here. Note that sensitivity for PCA is done
    ## on ALL available parameters, while for ppscore or pearson only on selected ones
    
    
    #%%
    
    ## Applying PCA. It is done late to preserve all attributes for sensitivity test
    if PCA_n != -1:
        scaler_pca = MinMaxScaler() # new scaler here because PCA can push 
                                    # variables out of (-1,1) bounds
                                    
        pca = PCA(n_components = PCA_n)
        
        X_train = pca.fit_transform(X_train)
        X_train = scaler_pca.fit_transform(X_train)
        
        X_test = pca.transform(X_test)
        X_test = scaler_pca.transform(X_test)
    
    
    
    
    
    #%%
    
    ## Data shaping
    
    ## from now on, arrays are being morphed into shapes valid for RNN+MLP
    
    memory = int(hMemoryMeters/step_length)
    
    imagination = int(imagination_meters/step_length)
    
    X_attr = list(X_train)
    
    try:
        X_train = X_train.to_numpy()    
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()    
    except:
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
    
    X_test = np.concatenate([X_train[-memory+1:,:], X_test], axis=0)
    y_test = np.concatenate([y_train[-memory+1:], y_test], axis=0)
    #%%
    
    def prepare(data, start, stop, cut_margin = 0, lcs=False):
        memory = stop-start
        stack = []
        for i in range(memory):
            stack.append(np.roll(data, -i))
    
        stack = np.flip(np.rot90(stack), axis=0)[start:-memory+1-cut_margin]
    
        if lcs == True:
            zero = stack[:,0]
    
            for j in range(len(zero)):
                stack[j] = stack[j] - zero[j]
        return stack
    
    target_lcs_correction = 1
    
    if target in lcs_list:
        X_train_RNN = prepare(np.squeeze(y_train), 0, memory, cut_margin = imagination, lcs=True)
        X_test_RNN = prepare(np.squeeze(y_test), 0, memory, cut_margin = imagination, lcs=True)
        
        y_train_RNN = prepare(np.squeeze(y_train), memory, memory+imagination, lcs=True)
        y_test_RNN = prepare(np.squeeze(y_test), memory, memory+imagination, lcs=True)
        
        offset_train = X_train_RNN[:,-1]
        offset_test = X_test_RNN[:,-1]
        
        for k in range(len(offset_train)):
            y_train_RNN[k] = y_train_RNN[k] + offset_train[k]
            
        for k in range(len(offset_test)):
            y_test_RNN[k] = y_test_RNN[k] + offset_test[k]
        
        target_lcs_correction = 1/np.max(y_train_RNN)
        
        y_train_RNN = y_train_RNN * target_lcs_correction
        X_train_RNN = X_train_RNN  * target_lcs_correction
        y_test_RNN = y_test_RNN * target_lcs_correction
        X_test_RNN = X_test_RNN * target_lcs_correction
        
    else:
        X_train_RNN = prepare(np.squeeze(y_train), 0, memory, cut_margin = imagination)
        X_test_RNN = prepare(np.squeeze(y_test), 0, memory, cut_margin = imagination)
        y_train_RNN = prepare(np.squeeze(y_train), memory, memory+imagination)
        y_test_RNN = prepare(np.squeeze(y_test), memory, memory+imagination)
    
    #%%
    
    
    
    
    X_train_MLP = []
    X_test_MLP = []
    
    #%%
    if PCA_n == -1:
        X_lcs_correction = [1]*len(X_train[0])
        
        for i in range(len(X_train[0])):
            if keep_columns[i] in lcs_list:
                X_train_MLP.append(prepare(X_train[:,i],0,memory+imagination, lcs=True))
                X_lcs_correction[i] = 1/np.max(X_train_MLP[i])
                X_train_MLP[i] = X_train_MLP[i]*X_lcs_correction[i]
            else:
                X_train_MLP.append(prepare(X_train[:,i],0,memory+imagination))
            
        X_train_MLP = np.asarray(X_train_MLP)
        X_train_MLP = np.concatenate(X_train_MLP[:,:, np.newaxis], axis = 1)
        X_train_MLP = np.rot90(X_train_MLP, axes=(1,2), k=3)
        
        
        
        
        
        for i in range(len(X_test[0])):
            if keep_columns[i] in lcs_list:
                X_test_MLP.append(prepare(X_test[:,i],0,memory+imagination, lcs=True))
                X_test_MLP[i] = X_test_MLP[i]*X_lcs_correction[i]
            else:
                X_test_MLP.append(prepare(X_test[:,i],0,memory+imagination))
            
        X_test_MLP = np.asarray(X_test_MLP)
        X_test_MLP = np.concatenate(X_test_MLP[:,:, np.newaxis], axis = 1)
        X_test_MLP = np.rot90(X_test_MLP, axes=(1,2), k=3)
        
    else:
        for i in range(len(X_train[0])):
            X_train_MLP.append(prepare(X_train[:,i],0,memory+imagination))
                
        X_train_MLP = np.asarray(X_train_MLP)
        X_train_MLP = np.concatenate(X_train_MLP[:,:, np.newaxis], axis = 1)
        X_train_MLP = np.rot90(X_train_MLP, axes=(1,2), k=3)
            
        for i in range(len(X_test[0])):
            X_test_MLP.append(prepare(X_test[:,i],0,memory+imagination))
                
        X_test_MLP = np.asarray(X_test_MLP)
        X_test_MLP = np.concatenate(X_test_MLP[:,:, np.newaxis], axis = 1)
        X_test_MLP = np.rot90(X_test_MLP, axes=(1,2), k=3)
    #%%
    
    X_train_RNN_m = X_train_RNN[:,:,np.newaxis]
    #X_train_MLP_m = X_train_MLP[:,:,np.newaxis]
    X_train_m = [X_train_RNN_m, X_train_MLP]#_m]
    
    X_test_RNN_m = X_test_RNN[:,:,np.newaxis]
    #X_test_MLP_m = X_test_MLP[:,:,np.newaxis]
    X_test_m = [X_test_RNN_m, X_test_MLP]#_m]
    #%%
        
        
        
    #%%
    
    ## Local coordinate system
    
    ## Just a cumsum on parameter converted to delta earlier
    
    
    #%%
    
    ## ML model definition
    
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import (Dense, Dropout, GRU, Flatten,
                                         GaussianNoise, concatenate, LSTM)
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import MaxPool1D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.callbacks import ModelCheckpoint
    import tensorflow as tf
    
    from tensorflow.keras.models import load_model
    
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    tf.keras.backend.clear_session()
    
    visible1 = Input(shape=(memory,1))
    
    
    visible2 = Input(shape=((memory+imagination),len(X_train[0])))
    
    
    
    x1 = GaussianNoise(0.02)(visible1)
    
    
    x1 = GRU(units=hGRU, kernel_initializer =  'glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer="zeros", kernel_regularizer='l2', recurrent_regularizer=None,
               bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
               recurrent_constraint=None, bias_constraint=None, return_sequences=False,
               return_state=False, stateful=False)(x1)
    
    x1 = Dropout(0.05)(x1)
    
    #x1 = Model(inputs=input1, outputs=x1)
    
     
    
    
    conv11 = Conv1D(hConv1, kernel_size=hConv1k, activation='relu')(visible2)
    pool11 = MaxPool1D(pool_size=hPoolSize)(conv11)
    drop2 = Dropout(0.05)(pool11)
    dense2 = Dense(hDense2)(drop2)
    # conv12 = Conv1D(16, kernel_size=4, activation='relu')(pool11)
    # pool12 = MaxPool1D(pool_size=2)(conv12)
    flat1 = Flatten()(drop2)
    #x2 = Dense(dense_units_1, kernel_initializer = 'glorot_uniform')(input2)
    #x2 = Dropout(0.05)(input2)#(x2)
    #x2 = Flatten()(x2)
    
    
    
    
    #x2 = Dense(1, kernel_initializer = 'glorot_uniform')(x2)
    
    #x2 = Model(inputs=input2, outputs=x2)
    
    combined = concatenate([x1, flat1])
    
    #z = Dense(1, activation="relu")(combined)
    z = Dense(imagination, activation="linear")(combined)
    
    model = Model(inputs=[visible1, visible2], outputs=z)
    
     
    
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
    
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss',
                                 mode='min', save_best_only=True, verbose=0)
    
    
       
    
    
    model.compile(optimizer='adam',loss='mean_squared_error')
    
    
    ## Training    
    history = model.fit(X_train_m,y_train_RNN,validation_data=(X_test_m,y_test_RNN),
                                epochs=2000, verbose=1, batch_size=8,
                                callbacks=[es, mc])
    
    model = load_model(f'best_model.h5')
    
    result = model.evaluate(X_test_m, y_test_RNN)
    
    if np.isnan(result):
        result = 0
        
    return(-np.log10(result))


## Plots
# pred = model.predict(X_train_m)

# for i in range(10):
#     s = np.random.randint(0, len(y_train_RNN))
    
#     x = np.arange(0,len(X_train_RNN[0]),1)
    
#     plt.title('Train')
#     plt.plot(x, X_train_RNN[s], label='RNN input')
    
#     x = np.arange(len(X_train_RNN[0]), len(X_train_RNN[0]) + len(y_train_RNN[0]),1)
#     plt.plot(x,y_train_RNN[s], label='RNN output, true')
    
    
    
#     plt.plot(x,pred[s], label='RNN output, predicted')
#     plt.legend()
    
    
#     plt.show()

# pred = model.predict(X_test_m)


# for i in range(5):
#     s = np.random.randint(0, len(y_test_RNN))
    
#     x = np.arange(0,len(X_test_RNN[0]),1)
    
#     plt.plot(x, X_test_RNN[s], label='RNN input')
    
#     x = np.arange(len(X_test_RNN[0]), len(X_test_RNN[0]) + len(y_test_RNN[0]),1)
#     plt.plot(x,y_test_RNN[s], label='RNN output, true')
    
    
#     plt.title('test')
#     plt.plot(x,pred[s], label='RNN output, predicted')
#     plt.legend()
#     plt.show()
#%%
# Tuning

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Bounded region of parameter space


pbounds =      {'hGRU' : (2, 512),
                'hAttrCount' : (1, 10),
                'hDense1' : (1, 256),
                'hMemoryMeters' : (1, 50),
                'hstep_extension' : (5, 20),
                'hConv1' : (1,32),
                'hConv1k' : (1,16),
                'hPoolSize' : (1,16),
                'hDense2' : (1,16)}



optimizer = BayesianOptimization(
    f=optimize_me,
    pbounds=pbounds,
    random_state=1,
)

logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=10,
    n_iter=1000,
)
#%%

# for col in list(dfs):
#     dfs[col].ffill().plot()
#     plt.title(col)
#     plt.show()
    