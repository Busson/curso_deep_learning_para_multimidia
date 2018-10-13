'''
"Desenvolvendo Modelos de Deep Learning para Aplicações Multimídia no Tensorflow" - Webmedia2018
Copyright (C) 2018 by Antonio J. Grandson Busson <busson@outlook.com>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import cv2
from draw_boxes import *


options = {"model": "cfg/yolo.cfg", 
           "load": "cfg/yolo.weights", 
           "threshold": 0.1}

tfnet = TFNet(options)

img = cv2.imread("images/sample1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)
print(result)

plt.imshow(boxing(img, result, 0.3))
plt.show()