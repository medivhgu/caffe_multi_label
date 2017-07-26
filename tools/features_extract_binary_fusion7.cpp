//#include <cuda_runtime.h>

#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/caffe.hpp"
//#include "caffe/blob.hpp"

using namespace std;
using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
	if (argc < 5) {
    LOG(ERROR) << std::endl
      << "extract-layer-features-from-images" << std::endl
      << "    net-deploy.proto.txt" << std::endl
      << "    pretrained-net-model.proto" << std::endl
      << "    CPU/GPU device-id" << std::endl
      << "    frame-dir file-list" << std::endl
      << "    output-file-base-name layers";
    return 0;
  }

  int device_id = 0;
  if (strcmp(argv[3], "GPU") == 0) {
    device_id = atoi(argv[4]);
    LOG(ERROR) << "Using GPU " << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else if (strcmp(argv[3], "CPU") == 0) {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  // Read model paramters
	boost::shared_ptr<Net<float> > caffe_test_net(
			new Net<float>(argv[1], caffe::TEST) );
	caffe_test_net->CopyTrainedLayersFrom(argv[2]);

  //const float mean_acquired[3] = {104, 117, 123};  

  // Read file list
  string frame_dir = argv[5];
  std::ifstream infile(argv[6]);
  std::vector<string> frame_names;
  std::vector<string> frame_files;
  string frame_name;
  while(getline(infile, frame_name)) {
    frame_names.push_back(frame_name);
    frame_files.push_back(frame_name);
  }

  // Read layers
  string output_file_base = argv[7];
  int n_layer = argc - 8;
  FILE *output_files[n_layer];
  string layers[n_layer];
  for (int i = 0; i < n_layer; i++) {
    layers[i] = argv[8 + i];
    for (int j = 0; j < strlen(argv[8 + i]); j++) {
			if (argv[8 + i][j] == '/') argv[8 + i][j] = '_';
		}
		string output_file_name = output_file_base + ".bin.trimean." + argv[8 + i];
    LOG(INFO) << "writing layer " << layers[i] << " into " << output_file_name << std::endl;
    output_files[i] = fopen(output_file_name.c_str(), "wb");
    const int frame_number = frame_files.size();
    const int feat_dim = caffe_test_net->blob_by_name(layers[i])->channels();
    fwrite(&frame_number, sizeof(int), 1, output_files[i]);
    fwrite(&feat_dim, sizeof(int), 1, output_files[i]);
  }

  cv::Mat cv_img, cv_img_orig;
  //int image_size = 256;
  int batch_size = caffe_test_net->blob_by_name("data")->num();
  int n_channel = caffe_test_net->blob_by_name("data")->channels();
	int vector_height = caffe_test_net->blob_by_name("data")->height();
	int vector_width = caffe_test_net->blob_by_name("data")->width();
	int n_image_crop_pixel = n_channel * vector_height * vector_width;
	LOG(INFO) << caffe_test_net->name();
  //LOG(INFO) << caffe_test_net->num_inputs() << ' ' << caffe_test_net->num_outputs();//TODO
  
	std::vector<string> batch_frame_names;
  std::vector<Blob<float>*> batch_frame_blobs;
  Blob<float>* frame_blob = new Blob<float>(batch_size, n_channel, vector_height, vector_width);
	cout << "frame_blob_channels: " << frame_blob->channels() << endl;
	cout << "frame_blob_height: " << frame_blob->height() << endl;
	cout << "frame_blob_width: " << frame_blob->width() << endl;
	cout << "n_image_crop_pixel: " << n_image_crop_pixel << endl;

  for (int frame_id = 0; frame_id < frame_files.size(); ++frame_id) {
    batch_frame_names.push_back(frame_names[frame_id]);
		/*******^_^*********/
		//read vector file to cv::Mat
		string filename = frame_dir + "/" + frame_files[frame_id];
		FILE* fd = fopen(filename.c_str(), "rb");
		float* data = new float[vector_width];
		fread(data, sizeof(float), vector_width, fd);
		fclose(fd);
		cv::Mat cv_img_tmp(1, vector_width, CV_32F, data);
		cv_img_tmp.copyTo(cv_img);
		delete[] data;
		/*******^_^*********/
    float* frame_blob_data = frame_blob->mutable_cpu_data();
    for (int i_channel = 0; i_channel < n_channel; ++i_channel) {
      for (int height = 0; height < vector_height; ++height) {
        for (int width = 0; width < vector_width; ++width) {
          int insertion_index = (frame_id % batch_size)* n_image_crop_pixel + (i_channel * vector_height + height) * vector_width + width;
          frame_blob_data[insertion_index] = (static_cast<float>(cv_img.at<cv::Vec3f>(height, i_channel)[width]) );
        }
      }
    }
    if ((frame_id + 1) % batch_size == 0 || (frame_id + 1) == frame_files.size()) {
      batch_frame_blobs.push_back(frame_blob);
      //const vector<Blob<float>*>& result = 
			caffe_test_net->Forward(batch_frame_blobs);

      const int n_example = batch_frame_names.size();
      for (int i_layer = 0; i_layer < n_layer; ++i_layer) {
        const shared_ptr<Blob<float> > data_blob = caffe_test_net->blob_by_name(layers[i_layer]);
        const float* data_blob_ptr = data_blob->cpu_data();
        const int n_channel = data_blob->channels();
				fwrite(data_blob_ptr, sizeof(float), n_example * n_channel, output_files[i_layer]);
				/*
				int* data_blob_int = new int[n_channel];
				for(int i=0; i < n_example; i++){ // the number of images
					//write the feature of a image
					for (int j=0; j < n_channel; j++) {
						data_blob_int[j] = (int)(data_blob_ptr[i * n_channel + j] * 10000);
					}
					fwrite(data_blob_int, sizeof(int), n_channel, output_files[i_layer]);
				}//fwrite
				*/
			}
      LOG(INFO) << frame_id + 1 << " (+" << n_example 
				<< ") out of " << frame_files.size() << " results written.";
      batch_frame_names.clear();
      batch_frame_blobs.clear();
    }
  }
  for (int i = 0; i < n_layer; i++) {
    fclose(output_files[i]);
  }
  delete frame_blob;
  return 0;
}
