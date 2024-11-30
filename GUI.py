from PyQt5 import QtWidgets, QtCore
import sys

def event(label_, click_time_):
    click_time_ = click_time_ + 1
    label_.setText(str(click_time_)) 
    

app = QtWidgets.QApplication(sys.argv)
window_main = QtWidgets.QMainWindow()
window_main.setObjectName("window_main")
window_main.setWindowTitle("test")
window_main.resize(1280, 720)

button_push = QtWidgets.QPushButton(window_main)
button_push.setGeometry(400, 200, 400, 200) # (x, y, w, h)
button_push.setObjectName("button")
button_push.setText("Add Number")
button_push.setStyleSheet('''
                            QPushButton
                            {
                                font-size:20px;
                                color: blue;
                                background: cyan;
                                border: 10px solid #000;
                            }
                            QPushButton:hover
                            {
                                color: yellow;
                                background: red;
                            }
                        ''')
                        
label = QtWidgets.QLabel(window_main)
label.setText('0')
label.setStyleSheet('''
                    QLabel
                    {
                        font-size:15px;
                        color: green;
                    }
                    ''')
label.setGeometry(200, 100, 200, 150)

click_time = 0

button_push.clicked.connect(lambda:event(label, click_time))



window_main.show()
sys.exit(app.exec_())