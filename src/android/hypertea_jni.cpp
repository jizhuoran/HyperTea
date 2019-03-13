/**
 * Original version of this file is provided in https://github.com/sh1r0/caffe,
 * which is part of https://github.com/sh1r0/caffe-android-lib.
 * Thanks to github user "sh1r0" for sharing this.
 */

#include <jni.h>

#include "android_gpu_net.hpp"

#ifdef __cplusplus
extern "C" {
#endif


JNIEXPORT jboolean JNICALL
Java_com_example_gsq_hypertea_1android_1project_HyperteaNet_creatNet(JNIEnv *env, jobject instance,
                                                     jstring weightPath_) {
    jboolean ret = true;

    const char *weightPath = env->GetStringUTFChars(weightPath_, 0);

    if (hypertea::new_net::get(weightPath) == NULL) {
        ret = false;
    }

    env->ReleaseStringUTFChars(weightPath_, weightPath);
    return ret;
}


JNIEXPORT jfloatArray JNICALL
Java_com_example_gsq_hypertea_1android_1project_HyperteaNet_inference(JNIEnv *env, jobject instance,
                                                  jfloatArray jinput_array) {


  std::vector<float> input_array, output_array;
  float * input_array_arr = (float *)env->GetFloatArrayElements(jinput_array, 0);
  input_array.assign(input_array_arr, input_array_arr+env->GetArrayLength(jinput_array));

  hypertea::new_net *hypertea_net = hypertea::new_net::get();
  
  hypertea_net->inference(input_array, output_array);

  // Handle result
  jfloatArray result = env->NewFloatArray(output_array.size());

  // move from the temp structure to the java structure
  env->SetFloatArrayRegion(result, 0, output_array.size(), output_array.data());
  return result;
}



JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env = NULL;
  jint result = -1;

  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    LOG(FATAL) << "GetEnv failed!";
    return result;
  }
  return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif