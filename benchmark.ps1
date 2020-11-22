$image_file = $args[0]
echo "==================================================="
echo "TFLite C++ CPU inference =========================="
echo "==================================================="
bazel-bin/example_classification/classification_cpu.exe --image_file=$image_file --number_of_inferences=100
echo "==================================================="
echo "TFLite C++ GPU inference =========================="
echo "==================================================="
bazel-bin/example_classification/classification_gpu.exe --image_file=$image_file --number_of_inferences=100
echo "==================================================="
echo "TFLite Python CPU inference ======================="
echo "==================================================="
python37 .\example_classification\example_classification.py --image_file=$image_file --number_of_inferences=100