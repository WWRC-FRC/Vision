"""
Lists the serial ports available on the
(Windows) computer.

Eli Bendersky (eliben@gmail.com)
License: this code is in the public domain
"""
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import serial
from serial.serialutil import SerialException
from serialutils import full_port_name, enumerate_serial_ports


class ListPortsDialog(QDialog):
    def __init__(self, parent=None):
        super(ListPortsDialog, self).__init__(parent)
        self.setWindowTitle('List of serial ports')

        self.ports_list = QListWidget()
        self.tryopen_button = QPushButton('Try to open')
        self.connect(self.tryopen_button, SIGNAL('clicked()'),
            self.on_tryopen)

        layout = QVBoxLayout()
        layout.addWidget(self.ports_list)
        layout.addWidget(self.tryopen_button)
        self.setLayout(layout)

        self.fill_ports_list()

    def on_tryopen(self):
        cur_item = self.ports_list.currentItem()
        if cur_item is not None:
            fullname = full_port_name(str(cur_item.text()))
            try:
                ser = serial.Serial(fullname, 38400)
                ser.close()
                QMessageBox.information(self, 'Success',
                    'Opened %s successfully' % cur_item.text())
            except SerialException, e:
                QMessageBox.critical(self, 'Failure',
                    'Failed to open %s:\n%s' % (
                        cur_item.text(), e))

    def fill_ports_list(self):
        for portname in enumerate_serial_ports():
            self.ports_list.addItem(portname)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = ListPortsDialog()
    form.show()
    app.exec_()

