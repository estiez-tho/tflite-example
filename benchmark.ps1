$image_file = $args[0]
bazel-bin/example_classification/classification.exe --image_file=$image_file --number_of_inferences=1000
bazel-bin/example_classification/classification.exe --inference_mode=GPU --image_file=$image_file --number_of_inferences=1000
python37 .\example_classification\example_classification.py --image_file=$image_file --number_of_inferences=1000