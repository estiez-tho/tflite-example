load("@rules_cc//cc:defs.bzl", "cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "classification",
    srcs=[
        "example_classification.cc"
    ], 
    deps = [
        "@windows_egl//:windows_egl_deps",
        "@windows_opencv//:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops_all_linked",
        "@org_tensorflow//tensorflow/lite/tools:gen_op_registration",
        "@org_tensorflow//tensorflow/lite/delegates/gpu:delegate",
        "@org_tensorflow//tensorflow/core:framework"
    ]
)
