export PYTHONPATH=/home/wuqianliang/facenet.git/trunk/src/
rm /home/wuqianliang/test/training-images_mtcnnpy_160 -frv
python src/align/findfaces_mtcnn.py /home/wuqianliang/test/training-images  /home/wuqianliang/test/training-images_mtcnnpy_160 --image_size 160 --margin 32 --gpu_memory_fraction 0.25
