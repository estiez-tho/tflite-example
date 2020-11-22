import tensorflow as tf
import numpy as np
import time
import sys
import argparse
import cv2

def run_inference(model_file, label_file, image_file, number_of_inferences):

    interpreter = tf.lite.Interpreter(model_path=model_file)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).astype(np.float32)
    image_height, image_width, _ = input_img.shape
    square_dim = min(image_height, image_width)
    delta_height = (image_height - square_dim) // 2
    delta_width = (image_width - square_dim) // 2
    cropped_img = input_img[delta_height:square_dim+delta_height, delta_width:square_dim+delta_width]

    input_shape = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    resized_img = cv2.resize(cropped_img, input_shape)

    interpreter.set_tensor(input_details[0]['index'], [resized_img])

    start = time.time()
    for i in range(number_of_inferences):
        interpreter.invoke()
    end = time.time()

    elapsed_seconds = end - start
    print()
    print(f"Ran {number_of_inferences} inferences in {elapsed_seconds}")
    print(f"Average time per inference : {1000 * elapsed_seconds / number_of_inferences} ms")
    print(f"Average number of inference per seconds : {number_of_inferences / elapsed_seconds}")


    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    with open(label_file, "r") as f:
        content = f.readlines()

    labels = [x.strip() for x in content] 



    score_labels = [(labels[i], scores[i]) for i in range(len(labels))]

    comp = lambda x : x[1]

    score_labels.sort(key = comp, reverse=True)
    print()
    print("Classification results : ")
    for i in range(5):
        print(f"Label : {score_labels[i][0]} - Score : {score_labels[i][1]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="./example_classification/mobilenet.tflite", help="TFLite model to use for the inference; Note : input must have RGB channels")
    parser.add_argument("--image_file", default="./images/goldfish.png", help="Image file to run the inference on")
    parser.add_argument("--label_file", default="./example_classification/labels.txt", help="File containing the labels for the classification")
    parser.add_argument("--number_of_inferences", default=1, help="number of inferences to run", type=int)
    args = parser.parse_args()
    print("Inference details : ")
    print(f"model_file : {args.model_file}")
    print(f"label_file : {args.label_file}")
    print(f"image_file : {args.image_file}")
    print(f"number_of_inference : {args.number_of_inferences}")
    run_inference(args.model_file, args.label_file, args.image_file, args.number_of_inferences)

if __name__ == "__main__":
    main()