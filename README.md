labelImg-YOLO
=============

![demo image](https://github.com/sawyre/labelImg-YOLO/blob/master/demo/demo1.jpg)

Необходимое програмное обеспечение
----------------------------------
Ссылка на исходный код приложения, так же там приведена инструкция по установке:
https://github.com/tzutalin/labelImg

Нужно, чтобы запустить приложение, но функция авторазметки будет заблокирована:
- Python 3
-	pip3
-	pyqt
-	lxml
## Установка под Ubuntu (Желательно 16.04)
1. sudo apt-get update
2. sudo apt-get install python3-pip
Python 3 + Qt5
3. sudo apt-get install pyqt5-dev-tools
4. sudo pip3 install lxml
5. make qt5py3 (Выполняется в директории приложения LabelImg)
6. python3 labelImg.py – запуск приложения из директории
Ссылка на исходный код YOLO2, так же там приведено необходимое ПО, для ее запуска:
https://github.com/experiencor/keras-yolo2
Нужно для запуска авторазметки:
### Для версии GPU:
-	Nvidia CUDA 8.0
-	CUDnn 6.0 for CUDA 8.0
-	Tensorflow-gpu 1.4.1
-	keras 2.1.5
-	opencv
-	imgaug
1. Свежий драйвер от NVIDIA
2. sudo pip3 install tensorflow-gpu == 1.4.1
Если при установке tensorflow установились CUDA и CUDnn, пункты 3 и 4 можно пропустить.
3. Nvidia CUDA 8.0
4. CUDnn 6.0 for CUDA 8.0 https://developer.nvidia.com/rdp/cudnn-archive#a-coll..
https://askubuntu.com/questions/767269/how-can-i-inst..
Если при запуске функции авторазметки в приложении будут проблемы с CUDnn, то:
4.1. gedit .bachrc
4.2. Добавить следующие строки в конец файла:+
-	export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
-	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
-	export PATH=/usr/local/cuda-8.0/bin:$PATH
5. sudo pip3 install keras == 2.1.5
6. sudo pip3 install opencv-python
При ошибке ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type помог пункт 6.1:
6.1. gedit .bachrc
И нужно закомментировать строчку source /opt/ros/kinetic/setup.bash
7. sudo pip3 install imgaug
### Для версии CPU:
-	tensorflow 1.4.1
-	keras 2.1.5
-	opencv
-	imgaug
1. sudo pip3 install tensorflow == 1.4.1
2. sudo pip3 install keras == 2.1.5
3. sudo pip3 install opencv-python
При ошибке ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type помог пункты 6.1:
3.1. gedit .bachrc 
3.2. Закоментировать строчку source /opt/ros/kinetic/setup.bash
4. sudo pip3 install imgaug

Руководство пользователя
------------------------
labelImg – программа для разметки изображений с помощью графического интерфейса. LabelImg позволяет проводить разметку в ручном и полуавтоматическом режиме. Полуавтоматический режим осуществляется за счет интегрированного в программу нейросетевого детектора на базе сети YOLO. Использование детектора позволяет заметно сократить время разметки за счет отображения предположений для выделения объектов. Данный полуавтоматический способ разметки является универсальным, поскольку можно использовать различные детекторы на базе YOLO, загрузив соответствующий файл конфигурации и файл с весовыми коэффициентами. 
## 1. Основные функции:
Open – открыть картинку.

Open Dir – открыть директорию с картинками

Change Save Dir – изменить директорию в которую сохраняются аннотации к картинкам.

Next Image – открыть следующую картинку из директории. Выбор конкретной картинки из директории также доступен в окне “File List” в 
правой панели, необходимо сделать двойной щелчок на имени файла.  

Prev Image – открыть предыдущую картинку из директории.

Next Box – выделить следующий отмеченный объект. Выбор объекта также доступен из окна “Box Labels” в правой панели.

Prev Box – выделить предыдущий отмеченный объект.

Save – сохранить аннотацию к картинке.

Create Rect Box – создать новую рамку объекта.

Paint Labels (View/Paint Labels)  - включить подписи классов.

predefined_classes.txt  (labelImg/data) – файл содержит классы определенные для ручной разметки. 

Autolabeling – Авторазметка (вкладка на верхней панели).

Set yolo images path – установить директорию, содержащую изображения для авторазметки.

Set yolo results path – установить директорию, для сохранения аннотаций сгенерированных нейронной сетью.

Set yolo config path – выбрать файл конфигурации YOLO.

Set yolo wheights path – выбрать файл содержащий весовые коэффициенты для YOLO.

Set yolo threshold – установить порог уверенности, ниже которого объекты не будут отображаться. ( threshold, от 0 до 1 )

Состояние параметров, необходимых для авторазметки, отображаются в окне правой панели “YOLO configuration” 

## 2. Таблица горячих клавиш

| Сочетание клавиш | Описание                                                          |
|:-----------------|:------------------------------------------------------------------|
| Ctrl+O           | Открыть картинку                                                  |
| Ctrl+U           | Открыть директорию с картинками                                   |
| Ctrl+R           | Изменить директорию для сохранения                                |
| Ctrl+shift+O     | Открыть папку с аннотациями                                       |
| Ctrl+S           | Сохранить аннотацию                                               |
| Ctrl+Shift+S     | Сохранить аннотацию в папке                                       |
| W                | Создать рамку объекта                                             |
| Ctrl+e           | Изменить класс выделенного объекта                                |
| N                | Выделить следующий созданный объект                               |
| B                | Выделить прошлый созданный объект                                 |
| D                | Следующая картинка                                                |
| A                | Предыдущая картинка                                               |
| Delete           | Удалить выделенный объект                                         |
| (number)         | Изменить класс выделенного объекта на класс под номером (number)  |
| Shift+Ctrl+s     | Выставить класс по умолчанию для всех выделяемых объектов         |
| Shift+Ctrl+p     | Показать подписи классов                                          |
| Shift+Ctrl+a     | Расширенная панель инструментов                                   |
