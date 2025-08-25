import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QLineEdit, QMessageBox
from ProcessCsv import *
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
from PyQt6.QtGui import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import re
from matplotlib.gridspec import GridSpec
import math

class target_window:
   def __init__(self, file_path, window_size_input):
      self.CSV_paths = file_path
      self.Window_size_input = window_size_input
      self.TargetWin = None


   def findratio(self, file_path):
      LenMin = 100000

      # num = len(file_path)

      for file in file_path:
         df = pd.read_csv(file)

         # 获取列名
         columns = df.columns.tolist()
         LenOfColumns = len(columns)
         if LenOfColumns < 100:
            # 代表旧版，有Power(W)，用分号做分隔符;
            df = pd.read_csv(file, sep=';')

         Time = pd.to_datetime(df.iloc[:, 0], format='%H:%M:%S')
         Sample_time = (Time[1] - Time[0]).seconds
         LenOfRows = len(df)
         if LenOfRows < LenMin:
            LenMin = LenOfRows
            Window_size = self.Window_size_input / Sample_time

      self.TargetWin = math.floor((LenMin - Window_size) / (Window_size * 0.01)) + 1
         # self.TargetWin = math.floor((self.LenMin - 60) / ( 60 * 0.1)) + 1

      return self.TargetWin

         # self.Overlap = 1 - (len(df) - self.Window_size) / ((self.TargetWin - 1) * self.Window_size)
         # step_size = int(self.Window_size * (1 - self.Overlap))  # 每次移动的步长


class data:
   def __init__(self, path, window_size_input, TargetWin, LaserSN):
      #self.Feature_List = None
      self.CSV_path = path
      #self.Power = None
      #self.Beam_x = None
      #self.Beam_y = None
      self.Inputset = None
      self.Sample_time = None
      self.Window_size_input = window_size_input
      self.Window_size = None
      self.Overlap = None
      self.Label = None
      self.Rows_of_CSV = None
      self.df_features = None
      self.Column_filtered_data = None
      self.Column_Row_filtered_data = None
      self.LenMin = None
      self.TargetWin = TargetWin
      self.Outputset = None
      self.List_old = [ ]
      self.LaserSN = LaserSN

   def init_data(self): # 从原始CSV整理出power, beam_x, beam_y, inputset, rows_of_csv, sample_time,label
      # 使用 Pandas 读取 CSV 文件
      df = pd.read_csv(self.CSV_path)

      # 获取列名
      columns = df.columns.tolist()
      LenOfColumns = len(columns)
      file_name = os.path.basename(self.CSV_path)
      if LenOfColumns < 100:
         # 代表旧版，有Power(W)，用分号做分隔符;
         self.List_old.append(self.CSV_path)
         print(f'{file_name} is recorded with old version')
         #df = pd.read_csv(self.CSV_path, sep=';')
         #columns = df.columns.tolist()
         #print(f'old columns: {columns}')
         #self.Power = df['Power (W)']
         #self.Beam_x = df[columns[65]]  # Beam Rad. X1/mm;
         #self.Beam_y = df[columns[66]]  # Beam Rad. Y1/mm;

         #self.Inputset = df[[columns[31], columns[33], columns[35], columns[40], columns[41], columns[34], columns[36],columns[37], columns[86], columns[84]]]
      else:
         # 代表新版，有Power meter CB(W)，用逗号做分隔符;
         #print(f'new columns: {columns}')
         #self.Power = df['Power meter CB(W)']
         #self.Beam_x = df['Beam Diameter X1(um)']  # Beam Rad. X1/mm;
         #self.Beam_y = df['Beam Diameter Y1(um)']  # Beam Rad. Y1/mm;

         self.Inputset = df[['Seeder Imon(A)', 'Preamplifier Imon(A)', 'Amplifier Imon(A)', 'Booster Imon(A)', 'SensorFLow(l/min)', 'SensorHumOpticsMainChamber1(%)', 'SensorT_OpeticsMainChamber1(°C)', 'SensorT_SeederHeatsink(°C)', 'SensorHumDiodeStack(%)', 'SensorT_DiodeStack(°C)', 'SensorHumLHC(%)', 'SensorT_LHC(°C)', 'SensorHumPSC(%)', 'SensorT_PSC(°C)', 'SensorHumFrqConvBox(%)', 'SensorT_FrqConvBox(°C)', 'Laser On', 'Modulator get(%)', 'Trigger mode', 'Actual frequency(kHz)', 'Water Temp. get(°C)']]

         self.Outputset =df[['Pulse Frequency(kHz)', 'Pulse Width(ns)', 'Pulse Amplitude(mV)', 'Pulse Area(nVs)', 'Power meter CB(W)', 'Beam Diameter X1(um)', 'Beam Diameter Y1(um)']]
         self.Outputset.replace(-np.inf, 0, inplace=True)

      # 获取文件名

      if "Abnormal" in file_name:  # 通过判断文件名称里有没有Abnormal来设置label为0或1
         self.Label = 0
         print("Abnormal Laser, label = 0")
      else:
         self.Label = 1
         print("Normal Laser, label = 1")

      self.Time = pd.to_datetime(df.iloc[:, 0], format='%H:%M:%S')
      self.Sample_time = (self.Time[1] - self.Time[0]).seconds
      self.Rows_of_CSV = len(df)
      #print('current length of feature Input is: ' + str(self.Rows_of_CSV) + ' rows')
      #print('current Input Sample time is: ' + str(self.Sample_time) + ' s') 经过检查，sample time 都是1s
      #这里可以return出 self.Sample_time


   def analyse(self): # 计算features_list

      #Input_list_Name = ['T_PSC', 'T_LD1', 'T_LD2', 'T_resonator', 'T_oven', 'H_LD2', 'H_resonator', 'H_oven', 'WaterFlow', 'T_water']
      new_dataframe_input_name_list = self.Inputset.columns.tolist()
      new_dataframe_output_name_list = self.Outputset.columns.tolist()
      LengthOfInput = len(self.Inputset.columns)
      LengthOfOutput = len(self.Outputset.columns)
      #LengthOfInput = len(Input_list_Name)

      #   初始化特征列表
      #features_names_list = ['mean_value', 'max_value', 'min_value', 'std_dev', 'median_value', 'corr_with_power', 'corr_with_BeamX', 'corr_with_BeamY']
      listOfDataframe = []

      #   先写列名
      for i in range(LengthOfInput):
         listOfDataframe.append(new_dataframe_input_name_list[i])
      for i in range(LengthOfOutput):
         listOfDataframe.append(new_dataframe_output_name_list[i])
      # for i in range(LengthOfInput):
      #    for j in range(LengthOfOutput):
      #       listOfDataframe.append('Corr_bewtween ' + new_dataframe_input_name_list[i] + ' and ' + new_dataframe_output_name_list[j])
      listOfDataframe.append('Label')
      listOfDataframe.append('LaserSN')
      self.df_features = pd.DataFrame(columns=listOfDataframe)

      # 定义时间窗口参数
      self.Window_size = int(self.Window_size_input / self.Sample_time)
      self.Overlap = float(1 - (self.Rows_of_CSV - self.Window_size) / ((self.TargetWin - 1) * self.Window_size))
      self.Overlap = round(self.Overlap, 2)
      step_size = int(self.Window_size * (1 - self.Overlap))  # 每次移动的步长
      if step_size == 0:
         step_size = 1

      feature_lines = 0
      for start_idx in range(0, self.Rows_of_CSV, step_size):
         end_idx = start_idx + self.Window_size

         # 如果窗口超出原 DataFrame 的长度，则跳过
         if end_idx > self.Rows_of_CSV:
            break

         new_dataframe_input_window = self.Inputset.iloc[start_idx:end_idx]
         new_dataframe_output_window = self.Outputset.iloc[start_idx:end_idx]
         #Power_window = self.Power.iloc[start_idx:end_idx]
         #BeamX_window = self.Beam_x.iloc[start_idx:end_idx]
         #BeamY_window = self.Beam_y.iloc[start_idx:end_idx]

         feature_list = []
         for i in range(LengthOfInput):
            mean_value_input = new_dataframe_input_window[new_dataframe_input_name_list[i]].mean()
            feature_list.append(mean_value_input)
            #max_value = new_dataframe_input_window[new_dataframe_input_name_list[i]].max()
            #min_value = new_dataframe_input_window[new_dataframe_input_name_list[i]].min()
            #std_dev = new_dataframe_input_window[new_dataframe_input_name_list[i]].std()
            #median_value = new_dataframe_input_window[new_dataframe_input_name_list[i]].median()
         for i in range(LengthOfOutput):
            mean_value_output = new_dataframe_output_window[new_dataframe_output_name_list[i]].mean()
            feature_list.append(mean_value_output)
            #corr_with_power = new_dataframe_input_window[new_dataframe_input_name_list[i]].corr(Power_window)
            #corr_with_BeamX = new_dataframe_input_window[new_dataframe_input_name_list[i]].corr(BeamX_window)
            #corr_with_BeamY = new_dataframe_input_window[new_dataframe_input_name_list[i]].corr(BeamY_window)
         # for i in range(LengthOfInput):
         #    for j in range(LengthOfOutput):
         #       corr_in_and_out = new_dataframe_input_window[new_dataframe_input_name_list[i]].corr(new_dataframe_output_window[new_dataframe_output_name_list[j]])
         #       feature_list.append(corr_in_and_out)

            #for j in range(len(features_names_list)):
               #feature_list.append(feature_values[j])

         feature_list.append(self.Label)  # 添加一列作为label
         feature_list.append(self.LaserSN)  # 添加一列作为LaserID

         self.df_features.loc[feature_lines] = feature_list
         feature_lines = feature_lines + 1

      self.df_features = round(self.df_features,3)
      self.df_features.fillna(0)

      return 1 #  Error code 1 = 没有问题


class Feature_Process:
   def __init__(self, file_path):
      path = file_path
      self.Cout = 0

   def drop_column_and_save(self, file_path, folder_path):
      #self.CsvPaths = []

      Feature_List = pd.read_csv(file_path)
      last_column = Feature_List.iloc[:, -1]
      Feature_List = Feature_List.drop(Feature_List.columns[-1], axis=1)
      variances = Feature_List.var()  # 计算每列的方差

      # 生成 0.001 到 0.01，每次递增 0.001
      first_part = pd.Series([0.001 * i for i in range(1, 11)])
      # 生成 0.02 到 0.1，每次递增 0.01
      second_part = pd.Series([0.01 * i for i in range(2, 11)])
      # 合并两个部分并转换成 DataFrame
      Variance = pd.DataFrame({'Variance': pd.concat([first_part, second_part], ignore_index=True)}).round(3)
      #CorValue = pd.DataFrame({'CorValue': [0.5, 0.9, 0.95]})
      CorValue = float(0.95)

      for value in Variance['Variance']:
         filtered_columns = variances[variances > value].index
         feature_droped_var = Feature_List[filtered_columns]  # 选择方差大于或等于输入的Variance的列
         CorMatrix = feature_droped_var.corr()
         CorMatrix = CorMatrix.round(6)  # 计算每两列间的相关系数，取6位小数

         #for Corvalue in CorValue['CorValue']:
         # 查找相关系数大于0.95 的列对（忽略对角线自身）
         ToDrop = []
         for i in range(len(CorMatrix.columns)):
            for j in range(i + 1, len(CorMatrix.columns)):
               if abs(CorMatrix.iloc[i, j]) >= CorValue:  # 选择两列之间相关系数大于输入的数字之中的一列删掉
                  col_to_remove = CorMatrix.columns[j]
                  if col_to_remove not in ToDrop:
                     ToDrop.append(col_to_remove)
         feature_droped = feature_droped_var.drop(columns=ToDrop)
         feature_droped["Label"] = last_column  # 在csv最后一列加Label
         self.Cout = self.Cout + 1

         DefaultFilename = 'TimeWindow60_Features_Var' + str(value) + '_Cor' + str(CorValue) + '.csv'
         default_path = os.path.join(folder_path, DefaultFilename)
         feature_droped.to_csv(default_path, index=False)
         #self.CsvPaths.append(default_path)
      return


class Learn_and_Draw:
   def __init__(self, file):
      self.path = file
      self.FigureFolder = r'C:\Users\maoya\Desktop\final edited\paired laser_weiter\Features\Droped'

   def learn_and_draw(self):
      df = pd.read_csv(self.path)

      # Calculate IQR
      Q1 = df.quantile(q=0.25, axis=0)
      Q3 = df.quantile(q=0.75, axis=0)
      IQR = Q3 - Q1
      # Determine the boundaries of outliers
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      outliers = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
      df = df[~outliers]

      # z-score标准化
      last_column = df.iloc[:, -1]
      last_column = last_column.reset_index(drop=True)  # 重置索引
      df = df.drop(df.columns[-1], axis=1)
      newcolumns = df.columns.tolist()
      ZScaler = StandardScaler()
      df_zscore = ZScaler.fit_transform(df)
      df = pd.DataFrame(df_zscore, columns=newcolumns)
      #  df["Label"] = last_column  # 在csv最后一列加Label
      df.loc[:, "Label"] = last_column  # 在csv最后一列加Label

      #Learning Part
      X = df.iloc[:, :-1]
      y = df.iloc[:, -1]
      feature_names = X.columns.tolist()
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

      # Train Random Forest
      rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
      rf_model.fit(X_train, y_train)
      rf_y_pred = rf_model.predict(X_test)
      rf_accuracy = accuracy_score(y_test, rf_y_pred)
      df.rf_Acc = rf_accuracy

      # Train SVM (Linear Kernel for coefficient analysis)
      svm_model = SVC(kernel='linear', C=1.0, gamma='scale')
      svm_model.fit(X_train, y_train)
      svm_y_pred = svm_model.predict(X_test)
      svm_accuracy = accuracy_score(y_test, svm_y_pred)
      df.svm_Acc = svm_accuracy

      # Train DNN (MLP)
      dnn_model = MLPClassifier(hidden_layer_sizes=(50,), activation="relu", solver="lbfgs", random_state=0, max_iter=1000)
      dnn_model.fit(X_train, y_train)
      dnn_y_pred = dnn_model.predict(X_test)
      dnn_accuracy = accuracy_score(y_test, dnn_y_pred)
      df.DNN_Acc = dnn_accuracy

      # ----------- Feature Importance Analysis -----------

      # Create subplots for all feature importance and accuracy
      fig1, axes1 = plt.subplots(2, 2, figsize=(32, 28))
      fig1.subplots_adjust(
         left=0.2,
         right=0.95,
         top=0.95,
         bottom=0.1,
         hspace=0.2,
         wspace=0.7
      )
      fig2, axes2 = plt.subplots(2, 2, figsize=(32, 28))
      fig2.subplots_adjust(hspace=0.4, wspace=0.8)
      fig5, axes5 = plt.subplots(2, 1, figsize=(32, 28))

      # 1. Random Forest Feature Importance (MDI)
      rf_importances = rf_model.feature_importances_
      rf_sorted_idx = np.argsort(rf_importances)
      axes1[0, 0].barh(np.array(feature_names)[rf_sorted_idx], rf_importances[rf_sorted_idx], color="skyblue")
      axes1[0, 0].set_xlabel("Random Forest Importance (MDI)", fontsize=35)
      axes1[0, 0].set_title("Feature Importance - Random Forest", fontsize=40)
      axes1[0, 0].tick_params(axis='y', labelsize=25, pad=10)
      axes1[0, 0].tick_params(axis='x', labelsize=35, pad=10)
      axes1[0, 0].set_xlim(0, rf_importances[rf_sorted_idx].max() * 1.5)

      # 2. Permutation Feature Importance (Random Forest)
      perm_importance_rf = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=0)
      perm_sorted_idx = np.argsort(perm_importance_rf.importances_mean)
      axes1[0, 1].barh(np.array(feature_names)[perm_sorted_idx], perm_importance_rf.importances_mean[perm_sorted_idx], color="coral")
      axes1[0, 1].set_xlabel("Permutation Importance (Random Forest)", fontsize=35)
      axes1[0, 1].set_title("Permutation Feature Importance - Random Forest", fontsize=35)
      axes1[0, 1].tick_params(axis='y', labelsize=25, pad=10)
      axes1[0, 1].tick_params(axis='x', labelsize=35, pad=10)
      axes1[0, 1].set_xlim(0, perm_importance_rf.importances_mean.max() * 1.5)


      # 3. SVM Feature Importance (Linear Kernel Coefficients)
      svm_importance = np.abs(svm_model.coef_).mean(axis=0)
      svm_sorted_idx = np.argsort(svm_importance)
      axes1[1, 0].barh(np.array(feature_names)[svm_sorted_idx], svm_importance[svm_sorted_idx], color="green")
      axes1[1, 0].set_xlabel("Feature Importance (SVM Coefficients)", fontsize=35)
      axes1[1, 0].set_title("Feature Importance - SVM (Linear Kernel)", fontsize=40)
      axes1[1, 0].tick_params(axis='y', labelsize=25, pad=10)
      axes1[1, 0].tick_params(axis='x', labelsize=35, pad=10)
      axes1[1, 0].set_xlim(0, svm_importance[svm_sorted_idx].max() * 1.5)

      # 4. Permutation Importance for DNN
      perm_importance_dnn = permutation_importance(dnn_model, X_test, y_test, n_repeats=10, random_state=0)
      perm_sorted_idx_dnn = np.argsort(perm_importance_dnn.importances_mean)
      axes1[1, 1].barh(np.array(feature_names)[perm_sorted_idx_dnn], perm_importance_dnn.importances_mean[perm_sorted_idx_dnn], color="purple")
      axes1[1, 1].set_xlabel("Permutation Importance (DNN)", fontsize=35)
      axes1[1, 1].set_title("Permutation Feature Importance - DNN", fontsize=40)
      axes1[1, 1].tick_params(axis='y', labelsize=25, pad=10)
      axes1[1, 1].tick_params(axis='x', labelsize=35, pad=10)
      axes1[1, 1].set_xlim(0, perm_importance_dnn.importances_mean[perm_sorted_idx_dnn].max() * 1.5)

      top10_rf = np.argsort(rf_importances)[-10:]
      top10_rf_perm = np.argsort(perm_importance_rf.importances_mean)[-10:]
      top10_svm = np.argsort(svm_importance)[-10:]
      top10_dnn_perm = np.argsort(perm_importance_dnn.importances_mean)[-10:]

      # 1. Random Forest Top 10
      axes2[0, 0].barh(
         np.array(feature_names)[top10_rf],
         rf_importances[top10_rf],
         color="skyblue"
      )
      axes2[0, 0].set_title("Top 10 Features - RF MDI", fontsize=35)
      axes2[0, 0].tick_params(axis='y', labelsize=25, pad=10)
      axes2[0, 0].tick_params(axis='x', labelsize=30)

      # 2. Permutation Importance RF Top 10
      axes2[0, 1].barh(
         np.array(feature_names)[top10_rf_perm],
         perm_importance_rf.importances_mean[top10_rf_perm],
         color="coral"
      )
      axes2[0, 1].set_title("Top 10 Features - RF Permutation", fontsize=35)
      axes2[0, 1].tick_params(axis='y', labelsize=25, pad=10)
      axes2[0, 1].tick_params(axis='x', labelsize=30)

      # 3. SVM Coefficients Top 10
      axes2[1, 0].barh(
         np.array(feature_names)[top10_svm],
         svm_importance[top10_svm],
         color="green"
      )
      axes2[1, 0].set_title("Top 10 Features - SVM Coefficients", fontsize=35)
      axes2[1, 0].tick_params(axis='y', labelsize=25, pad=10)
      axes2[1, 0].tick_params(axis='x', labelsize=30)

      # 4. DNN Permutation Top 10
      axes2[1, 1].barh(
         np.array(feature_names)[top10_dnn_perm],
         perm_importance_dnn.importances_mean[top10_dnn_perm],
         color="purple"
      )
      axes2[1, 1].set_title("Top 10 Features - DNN Permutation", fontsize=35)
      axes2[1, 1].tick_params(axis='y', labelsize=25, pad=10)
      axes2[1, 1].tick_params(axis='x', labelsize=30)

      axes5[0].barh(
         np.array(feature_names)[top10_rf],
         rf_importances[top10_rf],
         color="skyblue"
      )
      max_val_rf = rf_importances[top10_rf].max()
      axes5[0].set_xlim(0, max_val_rf * 1.5)
      axes5[0].set_title("Top 10 Features - RF MDI", fontsize=35)
      axes5[0].tick_params(axis='y', labelsize=25, pad=10)
      axes5[0].tick_params(axis='x', labelsize=30)

      axes5[1].barh(
         np.array(feature_names)[top10_svm],
         svm_importance[top10_svm],
         color="green"
      )
      max_val_svm = svm_importance[top10_svm].max()
      axes5[1].set_xlim(0, max_val_svm * 1.5)
      axes5[1].set_title("Top 10 Features - SVM Coefficients", fontsize=35)
      axes5[1].tick_params(axis='y', labelsize=25, pad=10)
      axes5[1].tick_params(axis='x', labelsize=30)

      plt.tight_layout()

      # 整理 Random Forest 的两种重要性
      rf_mdi_importance = pd.DataFrame({
         'Feature': feature_names,
         'RF_MDI_Importance': rf_importances
      })

      rf_perm_importance = pd.DataFrame({
         'Feature': feature_names,
         'RF_Permutation_Importance': perm_importance_rf.importances_mean
      })

      # SVM Importance
      svm_importance_df = pd.DataFrame({
         'Feature': feature_names,
         'SVM_Linear_Coef_Importance': svm_importance
      })

      # DNN Permutation Importance
      dnn_perm_importance_df = pd.DataFrame({
         'Feature': feature_names,
         'DNN_Permutation_Importance': perm_importance_dnn.importances_mean
      })

      # 合并所有 Feature Importance 到一个大表中
      importance_df = rf_mdi_importance.merge(
         rf_perm_importance, on='Feature'
      ).merge(
         svm_importance_df, on='Feature'
      ).merge(
         dnn_perm_importance_df, on='Feature'
      )
      importance_df = round(importance_df,5)

      filename_FeatureImportance = os.path.basename(self.path).replace('.csv', '_Feature Importance.csv')
      save_path_FeatureImportance = os.path.join(self.FigureFolder, filename_FeatureImportance)
      # 保存为 CSV 文件
      importance_df.to_csv(save_path_FeatureImportance, index=False)

      # ----------- Accuracy Comparison -----------
      fig3 = plt.figure(figsize=(16, 12))
      ax3 = fig3.add_subplot(111)
      models = ["Random Forest", "SVM", "DNN"]
      accuracies = [rf_accuracy, svm_accuracy, dnn_accuracy]
      bars = ax3.bar(models, accuracies, color=["blue", "green", "purple"])
      ax3.bar_label(bars, fmt='%.3f', padding=3, fontsize=40)
      ax3.set_ylim([0, 1.1])
      ax3.set_ylabel("Accuracy", fontsize=40)
      ax3.set_title("Model Accuracy Comparison", fontsize=40)
      ax3.tick_params(axis='both', labelsize=35)

      # Save the figure
      filename_analysis = os.path.basename(self.path).replace('.csv', '_Feature Importance.png')
      filename_analysis_top10 = os.path.basename(self.path).replace('.csv', '_Top10_Feature Importance.png')
      filename_analysis_top10_nur2 = os.path.basename(self.path).replace('.csv', '_Top10_Feature Importance_nur2.png')
      filename_comparison = os.path.basename(self.path).replace('.csv', '_Mode Comparison.png')
      self.FigureFolder = r'C:\Users\maoya\Desktop\final edited\paired laser_weiter\Features\Droped'
      if not os.path.exists(self.FigureFolder):
         os.makedirs(self.FigureFolder)
      save_path_analysis = os.path.join(self.FigureFolder, filename_analysis)
      save_path_analysis_top10 = os.path.join(self.FigureFolder, filename_analysis_top10)
      save_path_analysis_top10_nur2 = os.path.join(self.FigureFolder, filename_analysis_top10_nur2)
      save_path_comparison = os.path.join(self.FigureFolder, filename_comparison)
      fig1.savefig(save_path_analysis)
      fig2.savefig(save_path_analysis_top10)
      fig5.savefig(save_path_analysis_top10_nur2)
      fig3.savefig(save_path_comparison)
      plt.close('all')
      return


class MyWindow(QMainWindow, Ui_MainWindow):
   def __init__(self, parent=None):
      super(MyWindow, self).__init__(parent)
      self.setupUi(self)
      self.BtnLoad.clicked.connect(self.LoadCsv)
      self.BtnMerge.clicked.connect(self.MergeCsv)
      self.WinInputs.setValidator(QIntValidator())
      self.BtnDropColSave.clicked.connect(self.Column_Drop)
      self.BtnLearnAndDraw.clicked.connect(self.learn_and_draw)


   def LoadCsv(self):
      options = QFileDialog.Option.ReadOnly
      file_path, _ = QFileDialog.getOpenFileNames(
         self,
         "Select CSV file",
         r"C:\Users\maoya\Desktop",  # 修改为你的桌面路径
         "CSV Files (*.csv);;All Files (*)",
         options=options
      )

      if not file_path or file_path == ['']:  # 如果没有选择文件
         QMessageBox.information(self, "No File Selected", "No File Selected")
         return #避免程序卡住直接退出函数

      #Window_size = int(self.WinInputs.text()) / Sample_Time in Class data
      Window_size_input = int(self.WinInputs.text()) #这里应该处理一下sample time，因为check了sample time都是1所以没处理
      CsvFile = target_window(file_path, Window_size_input)
      CsvFile.findratio(file_path)

      FilePath = r'C:\Users\maoya\Desktop\final edited\paired laser_weiter\Features'
      if not os.path.exists(FilePath):
         os.makedirs(FilePath)

      for idx, path in enumerate(file_path):
         path_str = Path(path)
         LaserSN = path_str.stem.split("_")[0]
         FeatureList = data(path, Window_size_input, CsvFile.TargetWin, LaserSN)
         FeatureList.init_data()
         FeatureList.analyse()
         #filename = os.path.basename(path)
         if FeatureList.Label == 1:
            file_path_save = FilePath + f'\Feature_{LaserSN}_Normal.csv'
         else:
            file_path_save = FilePath + f'\Feature_{LaserSN}_Abormal.csv'
         FeatureList.df_features.to_csv(file_path_save, index=False)
         print(f'Lenght of old CSV file is : {len(FeatureList.List_old)}')

      print(str(len(file_path)) + 'CSV Files saved')
      QMessageBox.information(self, "Csv Saved", str(len(file_path)) + " Csv Document finished")

      return

   def MergeCsv(self):
      options = QFileDialog.Option.ReadOnly
      file_path, _ = QFileDialog.getOpenFileNames(
         self,
         "Select CSV file",
         r"C:\Users\maoya\Desktop",  # 修改为你的桌面路径
         "CSV Files (*.csv);;All Files (*)",
         options=options
      )
      if not file_path or file_path == ['']:  # 如果没有选择文件
         QMessageBox.information(self, "No File Selected", "No File Selected")
         return  # 避免程序卡住直接退出函数

      num = len(file_path)
      QMessageBox.information(self, "Csv files selected", str(num) + " Csv files selected")
      # Feature_List = pd.read_csv(file_path[0])
      # del(file_path[0])
      Feature_merged = pd.DataFrame()
      for file in file_path:
         Feature_merged = pd.concat([Feature_merged, pd.read_csv(file)])
         # self.Feature_List = pd.concat([self.Feature_List, pd.read_csv(file)])

      Feature_merged.fillna(0, inplace=True)
      Feature_merged = round(Feature_merged,3)

      # self.RowsOfFeature.setText(str(self.Feature_List.shape[0]))
      # self.ColsOfFeature.setText(str(self.Feature_List.shape[1]))
      FilePath = r'C:\Users\maoya\Desktop\final edited\paired laser_weiter\Features'
      file_path_save = FilePath + r'\Feature_Merged.csv'
      Feature_merged.to_csv(file_path_save, index=False)

      QMessageBox.information(self, "Csv files Saved", "Merged Csv file Saved")

      return

   def Column_Drop(self):
      options = QFileDialog.Option.ReadOnly
      file_path, _ = QFileDialog.getOpenFileName(
         self,
         "Select CSV file",
         r"C:\Users\maoya\Desktop",  # 修改为你的桌面路径
         "CSV Files (*.csv);;All Files (*)",
         options=options
      )

      if not file_path or file_path == ['']:  # 如果没有选择文件
         QMessageBox.information(self, "No File Selected", "No File Selected")
         return  # 避免程序卡住直接退出函数

      folder_path = r'C:\Users\maoya\Desktop\final edited\paired laser_weiter\Features\Droped'
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)
      drop_column = Feature_Process(file_path)
      drop_column.drop_column_and_save(file_path, folder_path)
      QMessageBox.information(self, "Csv files saved", str(drop_column.Cout) + " Csv files saved")


      return

   def learn_and_draw(self):
      options = QFileDialog.Option.ReadOnly
      file_path, _ = QFileDialog.getOpenFileNames(
         self,
         "Select CSV file",
         r"C:\Users\maoya\Desktop",  # 修改为你的桌面路径
         "CSV Files (*.csv);;All Files (*)",
         options=options
      )

      if not file_path or file_path == ['']:  # 如果没有选择文件
         QMessageBox.information(self, "No File Selected", "No File Selected")
         return  # 避免程序卡住直接退出函数

      QMessageBox.information(self, "CSV File Selected", f'{len(file_path)} CSV files selected')

      for file in file_path:
         Result = Learn_and_Draw(file)
         Result.learn_and_draw()

      QMessageBox.information(self, "Figure saved", f'{2*len(file_path)} Figures Saved')
      return


      #Window_size = int(self.WinInputs.text())



if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec())