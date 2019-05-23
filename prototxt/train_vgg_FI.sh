../caffe-master/build/tools/caffe train \
    --solver=vgg_solver_FI_c2.prototxt  \
    --weights=model/finetune_fi_vgg16_class2_v1_iter20000.caffemodel \
    -gpu 0 2>&1 | tee logs/finetune_FI_vgg16_class2_v1.txt
