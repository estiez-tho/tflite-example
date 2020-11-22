#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#ifdef USE_GPU
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif // USE_GPU
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

const std::string DEFAULT_MODEL = "./example_classification/mobilenet.tflite";
const std::string DEFAULT_IMAGE = "./images/goldfish.png";
const std::string DEFAULT_LABELS = "./example_classification/labels.txt";

ABSL_FLAG(std::string, model_file, DEFAULT_MODEL, "TFLite model to use for the inference; Note : input must have RGB channels");
ABSL_FLAG(std::string, image_file, DEFAULT_IMAGE, "Image file to run the inference on");
ABSL_FLAG(std::string, label_file, DEFAULT_LABELS, "File containing the labels for the classification");
ABSL_FLAG(int, number_of_inferences, 1, "Number of inference to run");

void ReadAndBindImageToTfInput(const char *image_file, float *input_tensor, TfLiteIntArray *dims)
{
    cv::Mat image = cv::imread(image_file);
    int image_width = image.size().width;
    int image_height = image.size().height;
    int square_dim = std::min(image_height, image_height);
    int delta_height = (image_height - square_dim) / 2;
    int delta_width = (image_width - square_dim) / 2;

    cv::Mat out_frame;
    // Crop the biggest centered square inside the image
    cv::resize(image(cv::Rect(delta_width, delta_height, square_dim, square_dim)), out_frame, cv::Size(dims->data[1], dims->data[2]));
    cv::Mat tensor_image(dims->data[1], dims->data[2], CV_32FC3, input_tensor);
    out_frame.convertTo(tensor_image, CV_32FC3);
    cv::cvtColor(tensor_image, tensor_image, cv::COLOR_BGR2RGB);
}

void ReadOutput(const char *label_file, float *output_tensor)
{
    std::ifstream in(label_file);
    std::string str;
    std::vector<std::string> labels;
    while (std::getline(in, str))
    {
        if (str.size() > 0)
            labels.push_back(str);
    }
    std::vector<std::pair<float, std::string>> ordered_scores;
    for (int i = 0; i < labels.size(); ++i)
    {
        float score = output_tensor[i];
        bool inserted = false;
        std::pair<float, std::string> current_pair{score, labels[i]};
        std::vector<std::pair<float, std::string>>::iterator it;
        for (it = ordered_scores.begin(); it != ordered_scores.end(); it++)
        {
            float score_it = (*it).first;
            if (score < score_it)
            {
                ordered_scores.insert(it, current_pair);
                inserted = true;
                break;
            }
        }
        if (!inserted)
            ordered_scores.emplace_back(current_pair);
    }

    std::reverse(ordered_scores.begin(), ordered_scores.end());

    std::cout << std::endl;
    std::cout << "Classification results : " << std::endl;
    for (int i = 0; i < 5; ++i)
    {

        std::cout << "Label : " << ordered_scores[i].second << " - Score : " << ordered_scores[i].first << std::endl;
    }
}

void run_inference(const char *image_file, const char *model_path, const char *label_file, int number_of_inferences)
{
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path);

    if (!model)
    {
        std::cout << "Could not build model from : " << model_path << std::endl;
        return;
    }
    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);

#ifdef USE_GPU
    auto *delegate = TfLiteGpuDelegateV2Create(nullptr);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        std::cout << "Could not setup GPU delegate" << std::endl;
        return;
    }
#endif // USE_GPU

    interpreter->AllocateTensors();
    auto input = interpreter->inputs()[0];

    TfLiteIntArray *dims = interpreter->tensor(input)->dims;

    // Read the image and emplace it in the model input
    ReadAndBindImageToTfInput(image_file, interpreter->typed_input_tensor<float>(0), dims);

    auto start = std::chrono::system_clock::now();
    // Run the inferences on the image
    for (int i = 0; i < number_of_inferences; ++i)
    {
        if (interpreter->Invoke() != kTfLiteOk)
        {
            std::cout << "Invoke failed" << std::endl;
            return;
        }
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << std::endl;
    std::cout << "Ran " << number_of_inferences << " inferences in " << elapsed_seconds.count() << std::endl
              << "Average time per inference : " << 1000 * elapsed_seconds.count() / number_of_inferences << " ms" << std::endl
              << "Average inference per seconds : " << number_of_inferences / elapsed_seconds.count() << std::endl;

    // Process the ouptput
    ReadOutput(label_file, interpreter->typed_output_tensor<float>(0));
}

int main(int argc, char **argv)
{
    /**
     * MUST HAVE THE 4 FOLLOWING ARGUMENTS : 
     * model_file : path to the tflite_model
     * label_file : path to the output labels
     * image_file : path to the image to run the inference on
    */
    absl::ParseCommandLine(argc, argv);
    std::string model_file = absl::GetFlag(FLAGS_model_file);
    std::string image_file = absl::GetFlag(FLAGS_image_file);
    std::string label_file = absl::GetFlag(FLAGS_label_file);
    int number_of_inferences = absl::GetFlag(FLAGS_number_of_inferences);

    std::cout
        << "Inference details : " << std::endl
        << "model_file : " << model_file << std::endl
        << "label_file : " << label_file << std::endl
        << "image_file : " << image_file << std::endl
        << "number_of_inference : " << number_of_inferences << std::endl;

    run_inference(image_file.c_str(), model_file.c_str(), label_file.c_str(), number_of_inferences);
}