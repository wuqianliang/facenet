rm /home/wuqianliang/test/training-images_mtcnnpy_160 -frv
python src/align/findfaces_mtcnn.py /home/wuqianliang/test/training-images  /home/wuqianliang/test/training-images_mtcnnpy_160 --gpu_memory_fraction 0.25
