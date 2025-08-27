import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QProgressBar, QLabel, QMessageBox, QLineEdit, QHBoxLayout, QTreeWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QTreeWidget
import sys
import shutil
import subprocess
import platform


class DetectFolder:
    def __init__(self,folder_path):
        self.path = folder_path
        self.foldername = os.path.basename(folder_path)
        self.LMTest = os.path.join(folder_path, 'LM-Test').replace('\\', r'/')

    def detectfolder(self):
        print(f'Current detecting Folder Name: {self.foldername}')
        print(f'Current detecting Folder Path: {self.path}')
        if not os.path.isdir(self.LMTest):
            print('No LM-Test inside')
            return

        case_folders = ['01_long time test', '02_beam profile', '03_pulse shape', '04_power', '05_internal camera',
                        '06_setup', '07_laser parameter', '08_checklist', '09_datasheet']

        for index, subfolder in enumerate(case_folders, start=0):
            Case_index = index
            subfolder_path = os.path.join(self.LMTest, subfolder)
            subfolder_path = subfolder_path.replace("\\", "/")
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                #   print(f'Folder Created: {subfolder_path}')
            #   else:
                #   print(f'Folder already existed: {subfolder_path}')

            case_folders[Case_index] = subfolder_path

        # 查找名字为 beamprofile（不区分大小写）
        target_folder = os.path.join(self.LMTest, '02_beam profile')
        source_folder = None
        for name in os.listdir(self.LMTest):
            if name.lower().strip() == '02_beamprofile':  #02_beam profile
                source_folder = os.path.join(self.LMTest, name)
                break

        if not source_folder or not os.path.isdir(source_folder):
            print(f"No '02_beamprofile' folder found in {self.LMTest}")

        # 移动文件
        if source_folder is not None and os.path.isdir(source_folder):
            for filename in os.listdir(source_folder):
                file_path = os.path.join(source_folder, filename)
                target_path = os.path.join(target_folder, filename)
                if os.path.isfile(file_path):
                    try:
                        shutil.copy2(file_path, target_path)  # 复制文件（包含元数据）
                        if os.path.getsize(file_path) == os.path.getsize(target_path):
                            os.remove(file_path)  # 确认复制成功后再删除原文件
                        else:
                            print(f"[!] Copy verification failed for {file_path}")
                    except Exception as e:
                        print(f"[!] Error copying {file_path} -> {e}")

            # 判断是否为空或只含 Thumbs.db
            remaining_files = [f for f in os.listdir(source_folder) if f.lower() != 'thumbs.db']
            if not remaining_files:
                thumbs_path = os.path.join(source_folder, 'Thumbs.db')
                if os.path.exists(thumbs_path):
                    try:
                        os.remove(thumbs_path)
                    except Exception as e:
                        print(f"Failed to delete Thumbs.db: {e}")
                try:
                    os.rmdir(source_folder)
                    print(f"Deleted empty folder: {source_folder}")
                except OSError as e:
                    print(f"Failed to delete {source_folder}: {e}")
            else:
                print(f"Folder not empty or doesn't exist: {source_folder}")
        #   all_failed = {}

        # --- 移动 03_pulse shape_trigger_HV 到 03_pulse shape ---
        target_folder_pulse = os.path.join(self.LMTest, '03_pulse shape')
        source_folder_pulse = None
        for name in os.listdir(self.LMTest):
            if name.lower().strip() == '03_pulse shape_trigger_hv':
                source_folder_pulse = os.path.join(self.LMTest, name)
                break

        if not source_folder_pulse or not os.path.isdir(source_folder_pulse):
            print(f"No '03_pulse shape_trigger_HV' folder found in {self.LMTest}")
        else:
            for filename in os.listdir(source_folder_pulse):
                file_path = os.path.join(source_folder_pulse, filename)
                target_path = os.path.join(target_folder_pulse, filename)
                if os.path.isfile(file_path):
                    try:
                        shutil.copy2(file_path, target_path)
                        if os.path.getsize(file_path) == os.path.getsize(target_path):
                            os.remove(file_path)
                        else:
                            print(f"[!] Copy verification failed for {file_path}")
                    except Exception as e:
                        print(f"[!] Error copying {file_path} -> {e}")

            # 判断是否为空或只含 Thumbs.db
            remaining_files = [f for f in os.listdir(source_folder_pulse) if f.lower() != 'thumbs.db']
            if not remaining_files:
                thumbs_path = os.path.join(source_folder_pulse, 'Thumbs.db')
                if os.path.exists(thumbs_path):
                    try:
                        os.remove(thumbs_path)
                    except Exception as e:
                        print(f"Failed to delete Thumbs.db: {e}")
                try:
                    os.rmdir(source_folder_pulse)
                    print(f"Deleted empty folder: {source_folder_pulse}")
                except OSError as e:
                    print(f"Failed to delete {source_folder_pulse}: {e}")
            else:
                print(f"Folder not empty or doesn't exist: {source_folder_pulse}")

        sn_failed_files = []
        for filename in os.listdir(self.LMTest):
            file_path = os.path.join(self.LMTest, filename)
            matched = False

            # 检查是否是文件（而非文件夹），且文件名包含"SN"并且扩展名为.pdf
            if os.path.isfile(file_path) and ('parameter' in filename or 'g0' in filename) and (filename.lower().endswith('.txt') or filename.lower().endswith('.csv')):
                target_path = os.path.join(case_folders[6], filename)
                matched = True
                #   print(f"File '{filename}' is moved to '{case_folders[6]}'")

            elif os.path.isfile(file_path) and ('screenshot' in filename.lower() or 'sreenshot' in filename.lower()) and (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
                target_path = os.path.join(case_folders[6], filename)
                matched = True
                #   print(f"File '{filename}' is moved to '{case_folders[6]}'")

            elif os.path.isfile(file_path) and ('checklist' in filename.lower()) and (filename.lower().endswith('.xls') or filename.lower().endswith('.xlsx') or filename.lower().endswith('.docx')):
                target_path = os.path.join(case_folders[7], filename)
                matched = True
                #   print(f"File '{filename}' is moved to '{case_folders[7]}'")

            elif os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
                target_path = os.path.join(case_folders[8], filename)
                matched = True
                #   print(f"File '{filename}' is moved to '{case_folders[8]}'")

            elif os.path.isfile(file_path) and ('internal camera' in filename) and (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
                target_path = os.path.join(case_folders[4], filename)
                matched = True
                #   print(f"File '{filename}' is moved to '{case_folders[4]}'")
            #   else:
                #   continue  # 不匹配任何类型则跳过
            if matched:
                try:
                    if not os.path.exists(target_path):
                        shutil.copy2(file_path, target_path)
                        if os.path.getsize(file_path) == os.path.getsize(target_path):
                            os.remove(file_path)
                        else:
                            print(f"[!] Copy verification failed for {file_path}")
                    #   print(f"File '{filename}' is moved to '{target_path}'")
                except Exception as e:
                    sn_failed_files.append(file_path)
                    print(f"Failed to move '{filename}' -> {e}")
            else:
                # 如果 file_path 在任意一个 case_folder 路径下，则跳过
                if any(os.path.commonpath([os.path.abspath(file_path), os.path.abspath(folder)]) == os.path.abspath(folder)for folder in case_folders):
                    continue  # 文件已经归入某个已知文件夹，跳过
                # 跳过 Thumbs.db 文件
                if os.path.basename(file_path).lower() == "thumbs.db":
                    continue

                sn_failed_files.append(file_path)

        if sn_failed_files:
            sn_name = os.path.basename(os.path.dirname(self.LMTest))
            return {sn_name: sn_failed_files}
        else:
            return {}  # 没有失败文件就返回空字典


class FolderWorker(QThread):
    progress_update = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)
    failed_files_signal = pyqtSignal(object)

    def __init__(self, folders, start_num, end_num):
        super().__init__()
        self.folders = folders
        self.start_num = start_num
        self.end_num = end_num

    def run(self):
        original_stdout = sys.stdout
        movement_name = f'Movement_{self.start_num}-{self.end_num}.txt'
        movement_path = os.path.join(r'D:/', movement_name)
        # 防止文件重名
        counter = 1
        while os.path.exists(movement_path):
            movement_path = os.path.join(r'D:/', f'Movement_{self.start_num}-{self.end_num}_{counter}.txt')
            counter += 1

        all_failed = {}
        with open(movement_path, 'w', encoding="utf-8") as file:
            sys.stdout = file
            for i, folder in enumerate(self.folders):
                folder_name = os.path.basename(folder)
                # 在此添加你的检测逻辑，例如检测特定文件
                detector = DetectFolder(folder)
                failed = detector.detectfolder()
                if failed:  # 合并到总失败列表
                    all_failed.update(failed)
                self.sleep(1)  # 模拟耗时操作
                self.progress_update.emit(i + 1, folder_name)
        sys.stdout = original_stdout
        self.failed_files_signal.emit(all_failed)  # 发送失败列表
        # 工作结束，发信号
        total = len(self.folders)
        self.finished_signal.emit(f"Detection succeeded.\n{total} folders checked.\nFile movement records saved.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.base_folder = ''
        self.setWindowTitle('Detect Multiple Folders')
        self.setGeometry(100, 100, 800, 600)

        self.button_select = QPushButton("Select Folder(parents)")
        self.button_select.clicked.connect(self.select_parent_folder)

        self.label_selected_folder = QLabel("No folder selected.")
        self.label_selected_folder.setStyleSheet("color: gray")

        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel('Standby')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText('Start SN number')

        self.end_input = QLineEdit()
        self.end_input.setPlaceholderText('End SN number')

        self.button_start = QPushButton('Start Detection')
        self.button_start.clicked.connect(self.start_detection)

        # Start SN 输入行
        start_label = QLabel('Start SN number: ')
        start_layout = QHBoxLayout()
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_input)

        # End SN 输入行
        end_label = QLabel('End SN number: ')
        end_layout = QHBoxLayout()
        end_layout.addWidget(end_label)
        end_layout.addWidget(self.end_input)

        # 添加 TreeWidget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(['Folder', ''])  # 第三个列用于按钮
        self.tree_widget.setColumnWidth(0, 250)

        layout = QVBoxLayout()
        layout.addWidget(self.button_select)
        layout.addWidget(self.label_selected_folder)
        layout.addLayout(start_layout)
        layout.addLayout(end_layout)
        #   layout.addWidget(self.start_input)
        #   layout.addWidget(self.end_input)
        layout.addWidget(self.button_start)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.tree_widget)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_parent_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folders')
        if not folder:
            QMessageBox.information(self, 'Info', 'No Folder Selected')
            return
        else:
            self.base_folder = folder.replace("\\", '/')
            self.label_selected_folder.setText(f'Folder Selected: {self.base_folder}')
            QMessageBox.information(self, 'Info', f'Folder Selected:\n{self.base_folder}')

    def start_detection(self):
        if not self.base_folder:
            QMessageBox.warning(self, 'Warning', 'Please select a parent folder first.')
            return

        self.tree_widget.clear()
        start_num = int(self.start_input.text())
        end_num = int(self.end_input.text())

        # 构造目标编号列表：['SN0000', 'SN0001', ..., 'SN0500']
        target_names = [f"SN{str(i).zfill(4)}" for i in range(start_num, end_num + 1)]

        # 获取该目录下所有子文件夹
        all_subfolders = [
            os.path.join(self.base_folder, name).replace('\\', '/')
            for name in os.listdir(self.base_folder)
            if os.path.isdir(os.path.join(self.base_folder, name)) and name.startswith("SN") and name in target_names
        ]
        if not all_subfolders:
            QMessageBox.information(self, 'Info', 'No matching folders found.')
            return
        #   print(all_subfolders)

        self.progress_bar.setMaximum(len(all_subfolders))
        self.worker = FolderWorker(all_subfolders, start_num, end_num)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished_signal.connect(self.show_message)  # 接收线程FolderWorker中发送的弹窗信号
        self.worker.start()
        self.worker.failed_files_signal.connect(self.display_failed_files)

    def update_progress(self, value, folder_name):
        self.progress_bar.setValue(value)
        self.status_label.setText(f'Now detecting: {folder_name}')

    def show_message(self, msg):
        QMessageBox.information(self, 'Movement Saved', msg)
        self.status_label.setText('Finished.')

    def display_failed_files(self, all_failed_dict):
        for folder_name, file_list in all_failed_dict.items():
            self.add_remaining_files_to_tree(folder_name, file_list)

    def add_remaining_files_to_tree(self, folder_name, file_list):
        folder_item = QTreeWidgetItem([folder_name])
        self.tree_widget.addTopLevelItem(folder_item)

        for file_path in file_list:
            file_item = QTreeWidgetItem([os.path.basename(file_path), '', ''])
            folder_item.addChild(file_item)

            open_button = QPushButton("Open")
            open_button_widget = QWidget()
            layout = QHBoxLayout(open_button_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(open_button)
            open_button.clicked.connect(lambda checked, path=file_path: self.open_file_location(path))

            self.tree_widget.setItemWidget(file_item, 1, open_button_widget)

    def open_file_location(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", f"The file does not exist:\n{path}")
            return

        if platform.system() == "Windows":
            # 选中文件而不是仅仅打开文件夹
            subprocess.run(['explorer', '/select,', os.path.normpath(path)])
        elif platform.system() == "Darwin":
            subprocess.run(['open', '-R', path])
        else:
            subprocess.run(['xdg-open', os.path.dirname(path)])

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

